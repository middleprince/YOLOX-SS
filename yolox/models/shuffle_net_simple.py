import torch
import torch.nn as nn
from torchstat import stat

from .network_blocks import Bottleneck       
from loguru import logger
from .shuffle_blocks import ShuffleV2Block
from ..utils import get_model_info

class ShuffleNetV2(nn.Module):
    def __init__(
            self,
            wid_mul=1,
            stage_blocks=(4, 8, 4),
            out_features=("shuffle3", "shuffle4", "shuffle5"),
            ):
        super(ShuffleNetV2, self).__init__()
        assert out_features, "please provide output faetures of ShuffleNet"
        self.out_features = out_features
    
        # Done: add width, depth weight as yolox do:
        # [256, 512, 1024] to match head pafpn
        # [64, 128, 256, 512, 1024], width weight is 0.375

        base_channels = int(wid_mul * 64) # 64   

        # done: design the repeat blocks in each stage
        self.stage_blocks = stage_blocks

        # stem /4 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            # add max pool to stem same as originalshufflenetV2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # stage2 / same 64
        self.stage2 = Bottleneck(
            int(base_channels), 
            int(base_channels),
            depthwise=True,
            act='relu'
        )     
        input_channel = base_channels 

        # /8-128, /16-256 /32-512
        stages = ["stage3", "stage4", "stage5"]

        # TODO: reformat the stage computation 
        for idxstage in range(len(self.stage_blocks)):
            stage_blocks = self.stage_blocks[idxstage]
            output_channel = input_channel * 2
            stage_features = []


            for i in range(stage_blocks):
                if i == 0:
                    stage_features.append(ShuffleV2Block(input_channel, output_channel, 
                                                mid_channels=output_channel // 2, ksize=3, stride=2))
                    logger.info(f"in index:{i} in repeat nums{stage_blocks}")
                    logger.info(f"in in_channel:{input_channel}, out_channel:{output_channel}")
                else:
                    stage_features.append(ShuffleV2Block(input_channel // 2, output_channel, 
                                                mid_channels=output_channel // 2, ksize=3, stride=1))

                input_channel = output_channel
            setattr(self, stages[idxstage], nn.Sequential(*stage_features))    

        self._initialize_weights()

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs['stem'] = x
         
        C2 = self.stage2(x)
        outputs['shuffle2'] = C2

        C3 = self.stage3(C2)
        outputs['shuffle3'] = C3

        C4 = self.stage4(C3)
        outputs['shuffle4'] = C4
        
        C5 = self.stage5(C4)
        outputs['shuffle5'] = C5

        return {k: v for k, v in outputs.items() if k in self.out_features}


    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    model = ShuffleNetV2(wid_mul=0.375)
    img_size=(224, 224)
    # logger.info("shufflenetV2 backbone Summary:{}".format(stat(model, (3, 320, 320))))
    logger.info("shufflenetV2 backbone Summary:")
    stat(model, (3, 224, 224))
    logger.info("shufflenetV2 backbone Summary:{}".format(get_model_info(model,img_size)))
    

    #print(model)

    #test_data = torch.rand(2, 3, 320, 320)
    #test_outputs = model(test_data)
    #for k, v in test_outputs.items():
    #    print(f"the stage name is: {k}", v.size())

