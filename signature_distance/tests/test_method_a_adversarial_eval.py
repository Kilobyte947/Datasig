import torch

from signature_distance.method_a_adversarial_eval import method_a_signature_distance


def test_method_a_signature_distance_zero_for_identical_positive_otherwise():
    torch.manual_seed(0)
    images = torch.rand(3, 28, 28)
    d_self = method_a_signature_distance(images, images)
    assert torch.allclose(d_self, torch.zeros(3), atol=1e-4)

    other = torch.rand(3, 28, 28)
    d_other = method_a_signature_distance(images, other)
    assert (d_other > 0).all()


def test_method_a_signature_distance_matches_default_r_from_sanity_check():
    # METHOD_A_R is hardcoded from run_experiment.sanity_check_demo's Phase 4
    # run (seed=0) rather than re-derived here - this just checks the two
    # don't silently drift apart.
    from signature_distance.data_pool import load_eval_pool
    from signature_distance.distances import choose_rescale_factor, rescale_signature
    from signature_distance.method_a_adversarial_eval import METHOD_A_PIXEL_ORDER, METHOD_A_R
    from signature_distance.signatures import signature_of_stream
    from signature_distance.streams import patch_sv_stream

    images, _ = load_eval_pool(n_per_class=30, seed=0)
    sig_raw = signature_of_stream(patch_sv_stream(images, METHOD_A_PIXEL_ORDER), depth=4)
    r = choose_rescale_factor(sig_raw, depth=4)
    assert abs(r - METHOD_A_R) < 1e-6
