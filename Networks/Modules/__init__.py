from Networks.Modules.GNNModules.EncodeProcessDecode import EncodeProcessDecode
from Networks.Modules.GNNModules.TSPModel import TSPModel
from Networks.Modules.GNNModules.TSPTransformer import TSPTransformer
from Networks.Modules.GNNModules.U_Net import UNet
from Networks.Modules.HeadModules.RLHead import RLHeadModuleTSP, RLHeadModule_agg_before, RLHeadModule_agg_after
from Networks.Modules.HeadModules.NormalHead import NormalHeadModule,TransformerHead
from Networks.Modules.HeadModules.ContinuousHead import ContinuousHead, ContinuousHeadChip
### TODO implement mixture of AnnealedNoise and Bernoulli Noise
GNNModel_registry = {"normal": EncodeProcessDecode, "TSPModel": TSPModel, "Transformer":TSPTransformer, "UNet": UNet}
OutputHead_registry = {"RLHead": RLHeadModule_agg_before, "RLHead_aggr": RLHeadModule_agg_after, "RLHeadTSP": RLHeadModuleTSP, "NormalHead": NormalHeadModule, "TransformerHead": TransformerHead, "ContinuousHead": ContinuousHead, "ContinuousHeadChip": ContinuousHeadChip}


def get_GNN_model(Model_name, train_mode, continuous_dim=0):
    """
    Get GNN model and appropriate output head.

    Args:
        Model_name: GNN architecture name
        train_mode: Training mode (PPO, REINFORCE, etc.)
        continuous_dim: If > 0, use continuous head for continuous outputs (e.g., chip placement)

    Returns:
        GNNModel, OutputHead
    """
    if(Model_name in GNNModel_registry.keys()):
        GNNModel = GNNModel_registry[Model_name]
    else:
        raise ValueError(f"GNN model {Model_name} is not implemented")

    # IMPORTANT: Check continuous_dim first!
    # For continuous problems (like chip placement), always use ContinuousHead
    if continuous_dim > 0:
        Head_name = "ContinuousHead"
        print(f"Using ContinuousHead for continuous_dim={continuous_dim}")
    # For discrete problems, select head based on train_mode and model
    elif(train_mode == "PPO"):
        if (Model_name == "Transformer"):
            Head_name = "RLHeadTSP"
        else:
            Head_name = "RLHead"
    else:
        if (Model_name == "Transformer"):
            Head_name = "TransformerHead"
        else:
            Head_name = "NormalHead"


    if(Head_name in OutputHead_registry.keys()):
        OutputHead = OutputHead_registry[Head_name]
    else:
        raise ValueError(f"OutputHead {Head_name} is not implemented")

    return GNNModel, OutputHead