from server.scenarios.generator import CommunityGenerator


def test_generator_task_sizes() -> None:
    g = CommunityGenerator()
    s1 = g.generate_task1(seed=1, n_accounts=90)
    assert s1.graph.number_of_nodes() > 80
    assert s1.trigger_post is not None

    s2 = g.generate_task2(seed=1, n_accounts=200)
    assert len(s2.coordinated_accounts) == 15
    assert len(s2.organic_harasser_accounts) == 40

    s3 = g.generate_task3(seed=1, n_accounts=400)
    assert len(s3.gateway_accounts) == 8
