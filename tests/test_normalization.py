import torch
from retrieval.gnn_model import GNNRetriever


def test_normalization_types():
    feature_size = 64
    hidden_size = 128
    num_layers = 2
    graph_dependencies = {"mode": "custom"}
    context_neighbor_verbosity = "clickable"
    edge_type_to_id = {"proof": 0}

    norm_types = ["none", "batch", "layer", "instance", "group"]

    for norm_type in norm_types:
        model = GNNRetriever(
            feature_size=feature_size,
            num_layers=num_layers,
            graph_dependencies=graph_dependencies,
            context_neighbor_verbosity=context_neighbor_verbosity,
            edge_type_to_id=edge_type_to_id,
            lr=1e-4,
            warmup_steps=100,
            gnn_layer_type="gcn",
            hidden_size=hidden_size,
            norm_type=norm_type,
        )

        if norm_type == "none":
            assert model.norm_layers is None
        else:
            assert model.norm_layers is not None
            assert len(model.norm_layers) == num_layers
            first_norm = model.norm_layers[0]
            if norm_type == "batch":
                assert isinstance(first_norm, torch.nn.BatchNorm1d)
            elif norm_type == "layer":
                assert isinstance(first_norm, torch.nn.LayerNorm)
            elif norm_type == "instance":
                assert isinstance(first_norm, torch.nn.InstanceNorm1d)
            elif norm_type == "group":
                assert isinstance(first_norm, torch.nn.GroupNorm)

        node_features = torch.randn(10, feature_size)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        with torch.no_grad():
            output = model.forward_embeddings(node_features, edge_index)

        assert output.shape == (10, feature_size)


def test_backward_compatibility():
    feature_size = 64
    graph_dependencies = {"mode": "custom"}
    context_neighbor_verbosity = "clickable"
    edge_type_to_id = {"proof": 0}

    # Verify batch norm behavior via the explicit new parameter
    model_old = GNNRetriever(
        feature_size=feature_size,
        num_layers=2,
        graph_dependencies=graph_dependencies,
        context_neighbor_verbosity=context_neighbor_verbosity,
        edge_type_to_id=edge_type_to_id,
        lr=1e-4,
        warmup_steps=100,
        norm_type="batch",
    )

    assert model_old.norm_layers is not None
    assert isinstance(model_old.norm_layers[0], torch.nn.BatchNorm1d)

    model_new = GNNRetriever(
        feature_size=feature_size,
        num_layers=2,
        graph_dependencies=graph_dependencies,
        context_neighbor_verbosity=context_neighbor_verbosity,
        edge_type_to_id=edge_type_to_id,
        lr=1e-4,
        warmup_steps=100,
        norm_type="layer",
    )

    assert model_new.norm_layers is not None
    assert isinstance(model_new.norm_layers[0], torch.nn.LayerNorm)


def test_gin_with_layer_norm():
    feature_size = 128
    hidden_size = 256

    model = GNNRetriever(
        feature_size=feature_size,
        num_layers=3,
        graph_dependencies={"mode": "custom"},
        context_neighbor_verbosity="clickable",
        edge_type_to_id={"proof": 0, "sig_lctx": 1},
        lr=1e-4,
        warmup_steps=100,
        gnn_layer_type="gin",
        hidden_size=hidden_size,
        norm_type="layer",
        use_residual=True,
        dropout_p=0.2,
    )

    assert model.norm_layers is not None
    assert len(model.norm_layers) == 3
    assert all(isinstance(layer, torch.nn.LayerNorm) for layer in model.norm_layers)

    node_features = torch.randn(20, feature_size)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)

    with torch.no_grad():
        output = model.forward_embeddings(node_features, edge_index)

    assert output.shape == (20, feature_size)
