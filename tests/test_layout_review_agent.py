from utils.layout_review_agent import (
    BlockSnapshot,
    LayoutReviewPlanner,
    ReviewModelConfig,
    ReviewAction,
    ExternalReviewProvider,
    HeuristicReviewProvider,
    collect_actions_from_list,
    collect_actions_by_type,
    filter_actions_by_block_indices,
    iterative_review,
)


def test_planner_proposes_fit_actions_for_overflow():
    planner = LayoutReviewPlanner()
    rst = planner.review_page(
        "p1",
        [
            BlockSnapshot(
                block_index=0,
                xyxy=(10, 10, 110, 50),
                text="A long sentence that should overflow this short box",
                font_size=28,
                est_text_size=(150, 70),
                bubble_center=(60, 30),
            )
        ],
    )
    actions = [a.action for a in rst.blocks[0].actions]
    assert "auto_fit" in actions
    assert "resize_to_fit_content" in actions


def test_planner_proposes_center_action_when_far_from_bubble_center():
    planner = LayoutReviewPlanner()
    rst = planner.review_page(
        "p1",
        [
            BlockSnapshot(
                block_index=1,
                xyxy=(0, 0, 100, 100),
                text="hello",
                font_size=18,
                est_text_size=(40, 20),
                bubble_center=(200, 200),
            )
        ],
    )
    actions = [a.action for a in rst.blocks[0].actions]
    assert "center_in_bubble" in actions


def test_collect_actions_by_type_groups_actions():
    planner = LayoutReviewPlanner()
    rst = planner.review_page(
        "p1",
        [BlockSnapshot(block_index=0, xyxy=(0, 0, 40, 40), text="x" * 30, font_size=20, est_text_size=(120, 80))],
    )
    grouped = collect_actions_by_type(rst)
    assert len(grouped["auto_fit"]) >= 1
    assert len(grouped["resize_to_fit_content"]) >= 1


def test_iterative_review_stops_when_apply_resolves_issues():
    start = [BlockSnapshot(block_index=0, xyxy=(0, 0, 80, 30), text="overflow", font_size=24, est_text_size=(120, 60))]

    def apply_actions(_actions):
        return [BlockSnapshot(block_index=0, xyxy=(0, 0, 180, 100), text="overflow", font_size=16, est_text_size=(90, 20))]

    result, loops = iterative_review("p1", start, apply_actions, max_iters=3)
    assert loops == 1
    assert result.blocks[0].actions == []


def test_collect_actions_from_list_groups_actions():
    grouped = collect_actions_from_list(
        [
            ReviewAction(action="auto_fit", block_index=0),
            ReviewAction(action="auto_fit", block_index=1),
            ReviewAction(action="resize", block_index=0, args={"scale": 1.1}),
        ]
    )
    assert len(grouped["auto_fit"]) == 2
    assert len(grouped["resize"]) == 1


def test_planner_tiny_dense_box_proposes_resize_scale():
    planner = LayoutReviewPlanner()
    rst = planner.review_page(
        "p1",
        [
            BlockSnapshot(
                block_index=4,
                xyxy=(0, 0, 48, 48),
                text="This is a dense sentence for a tiny area.",
                font_size=14,
                est_text_size=(42, 38),
            )
        ],
    )
    resize_actions = [a for a in rst.blocks[0].actions if a.action == "resize"]
    assert len(resize_actions) == 1
    assert resize_actions[0].args["scale"] > 1.0


def test_filter_actions_by_block_indices_keeps_only_requested_targets():
    actions = [
        ReviewAction(action="auto_fit", block_index=0),
        ReviewAction(action="center_in_bubble", block_index=2),
        ReviewAction(action="resize", block_index=5, args={"scale": 1.1}),
    ]
    filtered = filter_actions_by_block_indices(actions, [2, 5])
    assert [a.block_index for a in filtered] == [2, 5]


def test_heuristic_provider_returns_page_result():
    provider = HeuristicReviewProvider()
    cfg = ReviewModelConfig(provider="heuristic")
    result = provider.review("p1", [BlockSnapshot(block_index=0, xyxy=(0, 0, 30, 30), text="x", font_size=12)], cfg)
    assert result.page_name == "p1"
    assert len(result.blocks) == 1


def test_external_provider_uses_handler():
    def handler(page_name, blocks, config):
        assert config.provider == "cloud_api"
        return HeuristicReviewProvider().review(page_name, blocks, ReviewModelConfig(provider="heuristic"))

    provider = ExternalReviewProvider(handler)
    result = provider.review(
        "p2",
        [BlockSnapshot(block_index=1, xyxy=(0, 0, 40, 20), text="hello", font_size=10)],
        ReviewModelConfig(provider="cloud_api", prompt="custom"),
    )
    assert result.page_name == "p2"
