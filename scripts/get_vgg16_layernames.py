from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.feature_extraction import get_graph_node_names

vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
train_nodes, eval_nodes = get_graph_node_names(vgg)
print(eval_nodes[:20])