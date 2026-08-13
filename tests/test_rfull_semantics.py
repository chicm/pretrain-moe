import math
import unittest

import torch
import torch.nn.functional as F

from rfull_moe.semantics import (
    limited_swiglu,
    limited_swiglu_from_fused,
    load_balancing_loss_from_statistics,
    selected_topk_softmax,
    z_loss_from_statistics,
)


class LimitedSwiGLUTests(unittest.TestCase):
    def test_exact_asymmetric_clamps_and_gradient(self):
        gate = torch.tensor([-20.0, -3.0, 2.0, 11.0], requires_grad=True)
        up = torch.tensor([-12.0, -2.0, 3.0, 15.0], requires_grad=True)
        actual = limited_swiglu(gate, up)
        expected = F.silu(torch.minimum(gate, torch.tensor(10.0))) * torch.clamp(
            up, -10.0, 10.0
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        self.assertNotEqual(float(gate.grad[0]), 0.0)  # no lower gate clamp
        self.assertEqual(float(gate.grad[-1]), 0.0)  # upper gate clamp
        self.assertEqual(float(up.grad[0]), 0.0)
        self.assertEqual(float(up.grad[-1]), 0.0)

    def test_fused_layout_is_gate_then_up(self):
        fused = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        actual = limited_swiglu_from_fused(fused)
        expected = F.silu(torch.tensor([[1.0, 2.0]])) * torch.tensor([[3.0, 4.0]])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            limited_swiglu(torch.zeros(2), torch.zeros(3))
        with self.assertRaises(ValueError):
            limited_swiglu_from_fused(torch.zeros(2, 3))


class RouterEquationTests(unittest.TestCase):
    def test_selected_softmax_is_not_full_softmax_gather(self):
        logits = torch.tensor([[5.0, 4.0, 3.0, -1.0]], dtype=torch.bfloat16)
        weights, indices = selected_topk_softmax(logits, 2)
        expected = torch.softmax(torch.tensor([[5.0, 4.0]]), dim=-1)
        full_then_gather = torch.softmax(logits.float(), dim=-1).gather(-1, indices)
        torch.testing.assert_close(weights, expected, rtol=0, atol=0)
        self.assertFalse(torch.allclose(weights, full_then_gather))
        self.assertEqual(weights.dtype, torch.float32)
        self.assertEqual(indices.tolist(), [[0, 1]])
        torch.testing.assert_close(weights.sum(-1), torch.ones(1), rtol=0, atol=0)

    def test_aux_equation_and_gradient(self):
        counts = torch.tensor([3, 1, 0], dtype=torch.int64)
        sums = torch.tensor([1.2, 0.5, 0.3], dtype=torch.float64, requires_grad=True)
        loss = load_balancing_loss_from_statistics(
            counts, sums, token_count=2, topk=2, coefficient=1.0e-3
        )
        expected = 1.0e-3 * 3.0 * (
            (3.0 / 4.0) * (1.2 / 2.0)
            + (1.0 / 4.0) * (0.5 / 2.0)
            + (0.0 / 4.0) * (0.3 / 2.0)
        )
        self.assertAlmostEqual(float(loss), expected, places=9)
        loss.backward()
        expected_grad = torch.tensor(
            [1.0e-3 * 3.0 * 3.0 / 8.0, 1.0e-3 * 3.0 / 8.0, 0.0]
        )
        torch.testing.assert_close(sums.grad.float(), expected_grad, rtol=1e-6, atol=0)

    def test_ep_shards_must_combine_statistics_before_nonlinear_product(self):
        counts0 = torch.tensor([2, 0], dtype=torch.int64)
        counts1 = torch.tensor([0, 2], dtype=torch.int64)
        sums0 = torch.tensor([1.8, 0.2])
        sums1 = torch.tensor([0.6, 1.4])
        global_loss = load_balancing_loss_from_statistics(
            counts0 + counts1,
            sums0 + sums1,
            token_count=2,
            topk=2,
        )
        mean_local_scalar = 0.5 * (
            load_balancing_loss_from_statistics(
                counts0, sums0, token_count=1, topk=2
            )
            + load_balancing_loss_from_statistics(
                counts1, sums1, token_count=1, topk=2
            )
        )
        self.assertFalse(torch.isclose(global_loss, mean_local_scalar))

    def test_z_loss_equation(self):
        logits = torch.tensor([[1.0, 2.0], [-2.0, 0.0]], requires_grad=True)
        squared_sum = torch.square(torch.logsumexp(logits.float(), dim=-1)).sum()
        actual = z_loss_from_statistics(
            squared_sum, token_count=2, coefficient=1.0e-4
        )
        expected = 1.0e-4 * sum(
            math.log(sum(math.exp(float(x)) for x in row)) ** 2
            for row in logits.detach()
        ) / 2.0
        self.assertAlmostEqual(float(actual), expected, places=10)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_ep8_surrogate_matches_global_aux_and_z_router_gradients(self):
        torch.manual_seed(17)
        partitions = 8
        tokens_per_partition = 5
        experts = 11
        topk = 3
        aux_coeff = 1.0e-3
        z_coeff = 1.0e-4

        reference_logits = torch.randn(
            partitions * tokens_per_partition, experts, requires_grad=True
        )
        reference_probs = torch.softmax(reference_logits.float(), dim=-1)
        assignments = torch.topk(reference_logits.detach(), topk, dim=-1).indices
        global_counts = torch.bincount(
            assignments.reshape(-1), minlength=experts
        ).float()
        reference_aux = load_balancing_loss_from_statistics(
            global_counts,
            reference_probs.sum(dim=0),
            token_count=reference_logits.shape[0],
            topk=topk,
            coefficient=aux_coeff,
        )
        reference_z_sum = torch.square(
            torch.logsumexp(reference_logits.float(), dim=-1)
        ).sum()
        reference_z = z_loss_from_statistics(
            reference_z_sum,
            token_count=reference_logits.shape[0],
            coefficient=z_coeff,
        )
        (reference_aux + reference_z).backward()
        reference_grad = reference_logits.grad.detach().clone()

        local_logits = [
            part.detach().clone().requires_grad_(True)
            for part in reference_logits.detach().chunk(partitions, dim=0)
        ]
        local_probs = [torch.softmax(part.float(), dim=-1) for part in local_logits]
        detached_probability_sum = torch.stack(
            [probs.sum(dim=0).detach() for probs in local_probs]
        ).sum(dim=0)
        detached_z_sum = torch.stack(
            [
                torch.square(torch.logsumexp(part.float(), dim=-1)).sum().detach()
                for part in local_logits
            ]
        ).sum()
        global_tokens = partitions * tokens_per_partition
        true_aux = load_balancing_loss_from_statistics(
            global_counts,
            detached_probability_sum,
            token_count=global_tokens,
            topk=topk,
            coefficient=aux_coeff,
        )
        true_z = z_loss_from_statistics(
            detached_z_sum,
            token_count=global_tokens,
            coefficient=z_coeff,
        )

        corrected_losses = []
        for logits, probs in zip(local_logits, local_probs):
            aux_surrogate = load_balancing_loss_from_statistics(
                global_counts,
                probs.sum(dim=0),
                token_count=global_tokens,
                topk=topk,
                coefficient=aux_coeff * partitions,
            )
            local_z_sum = torch.square(
                torch.logsumexp(logits.float(), dim=-1)
            ).sum()
            z_surrogate = z_loss_from_statistics(
                local_z_sum,
                token_count=global_tokens,
                coefficient=z_coeff * partitions,
            )
            corrected_losses.append(
                aux_surrogate
                + (true_aux - aux_surrogate.detach())
                + z_surrogate
                + (true_z - z_surrogate.detach())
            )

        for corrected in corrected_losses:
            self.assertAlmostEqual(
                float(corrected.detach()),
                float((reference_aux + reference_z).detach()),
                places=8,
            )
        # Dense DDP averages replicated router gradients across the eight EP ranks.
        (sum(corrected_losses) / partitions).backward()
        observed_grad = torch.cat([part.grad for part in local_logits], dim=0)
        torch.testing.assert_close(observed_grad, reference_grad, rtol=2e-6, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
