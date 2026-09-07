import torch
from loguru import logger
from torch import nn

device = torch.device("cuda")


class Softmax_classify:
    def __init__(self, n_inputs, n_outputs, lr=0.1):
        # n_input = 28 * 28 (一张图片的像素点), n_output = 10
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(n_inputs),
            nn.Linear(n_inputs, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.33),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_outputs),
        ).to(device)
        # self.model = nn.Sequential(nn.Flatten(), nn.Linear(n_inputs, n_outputs)).to(
        #     device
        # )
        # self.optimizor = torch.optim.SGD(
        #     self.model.parameters(), lr=lr, weight_decay=0.01
        # )
        self.optimizor = torch.optim.AdamW(
            self.model.parameters(), lr, (0.9, 0.999), weight_decay=0.05
        )
        self.loss_fn = nn.CrossEntropyLoss()

    # 实际不需要
    def loss_entroy(self, predict, y):
        indices = torch.arange(len(y))
        return -torch.log(predict[indices, y]).mean()

    def forward(self, X):
        X = X.to(device)
        return self.model(X)

    def predict(self, X):
        with torch.no_grad():
            X.to(device)
            res = self.forward(X)
            return res.argmax(dim=1)

    # @log_train
    # @monitor(
    #     " epoch = {epoch} ,loss = {loss} , accuracy = {accuracy:.2f}",
    #     interval=1,
    #     console=True,
    # )
    def fit(self, train_data, epochs=2000, print_every=1):
        # train_info = []
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizor, epochs
        )
        logger.add("./log/train.log")
        self.model.train()
        for epoch in range(epochs):
            total, correct = 0, 0
            for X, y in train_data:
                X = X.to(device)
                y = y.to(device)
                pred = self.forward(X)
                loss = self.loss_fn(pred, y)
                self.optimizor.zero_grad()
                loss.backward()
                self.optimizor.step()

                total += len(y)
                correct += (pred.argmax(dim=1) == y).sum().item()
            self.scheduler.step()

            accuracy = correct / total

            if epoch % print_every == 0:
                logger.info(f"epoch {epoch} | loss = {loss} | accuracy = {accuracy}")

        self.model.eval()

        # return losses
