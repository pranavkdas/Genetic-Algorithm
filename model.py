class FaceNet(nn.Module,):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.model_ = InceptionResnetV1(pretrained='vggface2',classify=False)
        self.model = nn.Sequential(*list(self.model_.children())[:-5])
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.sec_last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
        self.last_linear = nn.Linear(512, 512, bias=False)
        self.logits = nn.Linear(512, 136)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = F.relu(self.sec_last_linear(x.view(x.shape[0], -1)))
        x = self.last_bn(x)
        x = F.relu(self.last_linear(x))
        x = self.logits(x)
        return x


net = FaceNet()
