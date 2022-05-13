from protors.protors import ProtoRS
from util.net import get_network
from util.args import get_args
from torchinfo import summary


num_channels = 3
args = get_args()
feature_net, add_on_layers = get_network(num_channels, args)
model = ProtoRS(200, feature_net, args, add_on_layers)
summary(model, input_size=(args.batch_size, 3, 224, 224))

