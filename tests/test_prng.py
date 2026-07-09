from sim_core import new_rng, next_u64, rand_below


def test_prng_deterministic():
    a = new_rng(42)
    b = new_rng(42)
    assert next_u64(a) == next_u64(b)


def test_prng_differs_by_seed():
    assert next_u64(new_rng(1)) != next_u64(new_rng(2))


def test_rand_below_in_range():
    rng = new_rng(7)
    for _ in range(1000):
        v = rand_below(rng, 10)
        assert 0 <= v < 10
