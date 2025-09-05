import torch
from retrieval.gnn_model import GNNRetriever


def test_enhanced_gin():
    """Test GIN with enhanced features."""

    feature_size = 128
    hidden_size = 256
    num_layers = 3
    graph_dependencies = {"mode": "custom", "use_proof_dependencies": True}
    context_neighbor_verbosity = "clickable"
    edge_type_to_id = {"proof": 0, "signature_clickable_lctx": 1, "signature_clickable_goal": 2}

    model = GNNRetriever(
        feature_size=feature_size,
        num_layers=num_layers,
        graph_dependencies=graph_dependencies,
        context_neighbor_verbosity=context_neighbor_verbosity,
        edge_type_to_id=edge_type_to_id,
        lr=1e-4,
        warmup_steps=100,
        gnn_layer_type="gin",
        hidden_size=hidden_size,
        use_residual=True,
        dropout_p=0.3,
        norm_type="layer",
        use_initial_projection=True,
    )

    # Verify components exist
    assert model.initial_projection is not None
    assert model.residual_projection is not None
    assert model.final_projection is not None
    assert model.norm_layers is not None
    assert len(model.norm_layers) == num_layers


def test_different_configurations():
    """Test different GNN configurations and a basic forward pass."""

    configurations = [
        {
            "name": "Standard GCN",
            "gnn_layer_type": "gcn",
            "hidden_size": None,
            "use_residual": False,
            "norm_type": "none",
            "use_initial_projection": False,
        },
        {
            "name": "RGCN with layer norm",
            "gnn_layer_type": "rgcn",
            "hidden_size": 64,
            "use_residual": True,
            "norm_type": "layer",
            "use_initial_projection": True,
        },
        {
            "name": "GAT with residuals",
            "gnn_layer_type": "gat",
            "hidden_size": None,
            "use_residual": True,
            "norm_type": "none",
            "use_initial_projection": True,
        },
    ]

    feature_size = 64
    graph_dependencies = {"mode": "custom"}
    context_neighbor_verbosity = "clickable"
    edge_type_to_id = {"proof": 0}

    for config in configurations:
        model = GNNRetriever(
            feature_size=feature_size,
            num_layers=2,
            graph_dependencies=graph_dependencies,
            context_neighbor_verbosity=context_neighbor_verbosity,
            edge_type_to_id=edge_type_to_id,
            lr=1e-4,
            warmup_steps=100,
            gnn_layer_type=config["gnn_layer_type"],
            hidden_size=config["hidden_size"],
            use_residual=config["use_residual"],
            dropout_p=0.2,
            norm_type=config["norm_type"],
            use_initial_projection=config["use_initial_projection"],
        )

        # Test forward pass
        node_features = torch.randn(10, feature_size)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        with torch.no_grad():
            output = model.forward_embeddings(node_features, edge_index)

        assert output.shape[0] == 10
        assert output.shape[1] == feature_size


def test_parameter_validation():
    """Test that different parameter combinations behave as expected."""

    feature_size = 128
    graph_dependencies = {"mode": "custom"}
    context_neighbor_verbosity = "clickable"
    edge_type_to_id = {"proof": 0}

    # hidden_size == feature_size (should not create projection layers)
    model1 = GNNRetriever(
        feature_size=feature_size,
        num_layers=2,
        graph_dependencies=graph_dependencies,
        context_neighbor_verbosity=context_neighbor_verbosity,
        edge_type_to_id=edge_type_to_id,
        lr=1e-4,
        warmup_steps=100,
        hidden_size=feature_size,
        use_residual=True,
        use_initial_projection=False,
    )

    assert model1.residual_projection is None
    assert model1.final_projection is None
    assert model1.initial_projection is None

    # Different hidden_size (should create projection layers)
    model2 = GNNRetriever(
        feature_size=feature_size,
        num_layers=2,
        graph_dependencies=graph_dependencies,
        context_neighbor_verbosity=context_neighbor_verbosity,
        edge_type_to_id=edge_type_to_id,
        lr=1e-4,
        warmup_steps=100,
        hidden_size=feature_size * 2,
        use_residual=True,
        use_initial_projection=True,
    )

    assert model2.residual_projection is not None
    assert model2.final_projection is not None
    assert model2.initial_projection is not None
