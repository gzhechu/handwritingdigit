#!/usr/bin/env python

import tkinter as tk
from tkinter import ttk
from math import hypot
import numpy as np
import torch
from torch import nn


# 定义颜色、网格尺寸和画笔尺寸
GRID_SIZE = 28
CELL_SIZE = 8
BRUSH_RADIUS = int(1.5 * CELL_SIZE)  # 笔触的半径
POINT_RADIUS = int(0.5 * CELL_SIZE)  # 笔尖的半径，
print(f"{CELL_SIZE} {BRUSH_RADIUS} {POINT_RADIUS}")

TAG_BRUSH = "BRUSH"
TAG_NUM = "NUMBER"
TAG_LINE = "LINE"
TAG_BAR = "BAR"

GRAYSCALE_RANGE = 255  # 最大灰度值

# 存储绘制网格的灰度值
grid_cells_value = {}
# 记录各层神经元的激活值
activation_values = []

olwn = None  # output layer weight normalized value
slwn = None  # second layer weight mormalized value


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Define model 创建模型，
# 我们训练的是一个有两个隐藏层的神经网络，每个隐藏层有 64 个神经元
# 请阅读  train_with_data.pdf 或 train_with_data.ipynb
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# 初始化并载入提前训练好的模型，目前载入的模型准确度 95%
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model_predict.pth"))
model.eval()


def get_gray_distance(dist: float, brush_r: float, point_r: float):
    # 计算灰度值，考虑笔触及笔尖的半径
    # print(f"value={value}, max_dist={max_dist}")
    v = min(
        int((brush_r - dist + point_r) * GRAYSCALE_RANGE / brush_r),
        GRAYSCALE_RANGE,
    )
    if v < 0:
        v = 0
    if v > GRAYSCALE_RANGE:
        v = GRAYSCALE_RANGE
    return v


def move_brush(event: tk.Event):
    draw_handwriting(event.x, event.y)
    on_predict()


def draw_handwriting(ev_x: int, ev_y: int):
    # 绘制笔迹
    min_x = max(0, ev_x - BRUSH_RADIUS)
    max_x = min(GRID_SIZE * CELL_SIZE - 1, ev_x + BRUSH_RADIUS)
    min_y = max(0, ev_y - BRUSH_RADIUS)
    max_y = min(GRID_SIZE * CELL_SIZE - 1, ev_y + BRUSH_RADIUS)
    # print(f"min_x: {min_x}, min_y: {min_y}, max_x: {max_x}, max_y: {max_y}")

    # print("{" * 32)
    # 遍历受影响的网格并计算灰度值，然后绘制
    for i in range(min_x, max_x + CELL_SIZE, CELL_SIZE):
        for j in range(min_y, max_y + CELL_SIZE, CELL_SIZE):
            px = i // CELL_SIZE
            py = j // CELL_SIZE
            if px >= GRID_SIZE:
                return
            if py >= GRID_SIZE:
                return

            dx = abs(px * CELL_SIZE + CELL_SIZE / 2 - ev_x)
            dy = abs(py * CELL_SIZE + CELL_SIZE / 2 - ev_y)
            dist = hypot(dx, dy)
            # print(
            #     f"i={i}, j={j}, dx={dx}, dy={dy}, dist={dist}, half_brush_size={half_brush_size}"
            # )
            gray_value = 0
            if dist <= BRUSH_RADIUS:
                gray_value = get_gray_distance(dist, BRUSH_RADIUS, POINT_RADIUS)

            old_value = grid_cells_value[(px, py)]
            gray_value = max(gray_value, old_value)
            if gray_value > 0:
                color_code = "#%02x%02x%02x" % (
                    GRAYSCALE_RANGE - gray_value,
                    GRAYSCALE_RANGE - gray_value,
                    GRAYSCALE_RANGE - gray_value,
                )

                # 绘制并记录网格对应颜色
                writingCanvas.create_rectangle(
                    px * CELL_SIZE + 1,
                    py * CELL_SIZE + 1,
                    (px + 1) * CELL_SIZE,
                    (py + 1) * CELL_SIZE,
                    fill=color_code,
                    width=0,
                    tag=TAG_BRUSH,
                )
                # print(f"save {gray_value} to {px} {py} near {grid_x} {grid_y}")
                grid_cells_value[(px, py)] = gray_value
    # print("}" * 32)


def reset_canvas():
    # 重置画图区
    label_predict = infoCanvas.find_withtag(TAG_NUM)
    infoCanvas.itemconfigure(label_predict, text="X")

    lines = networkCanvas.find_withtag(TAG_LINE)
    for l in lines:
        networkCanvas.delete(l)

    bars = networkCanvas.find_withtag(TAG_BAR)
    for bar in bars:
        networkCanvas.delete(bar)
    # 清除已绘制的所有网格
    cells = writingCanvas.find_withtag(TAG_BRUSH)
    for cell in cells:
        writingCanvas.delete(cell)

    # 更新整个画布
    reset_painted_cell()


def get_gray_value():
    output = []
    for cell, color_code in grid_cells_value.items():
        gray_value = color_code
        output.append(gray_value)
    val = np.array(output).reshape(28, 28)
    val[val > 0] = 1
    print(val)
    print("")
    return np.array(output).reshape(28, 28)


def on_predict():
    output = []
    for cell, color_code in grid_cells_value.items():
        gray_value = color_code
        output.append(gray_value)
    # print(output, len(output))
    x = torch.FloatTensor(output).reshape(1, 28, 28)
    x = (x) / GRAYSCALE_RANGE
    x = x.to(device)
    pred = model(x)
    get_activation_values(x)
    redraw_lines()
    redraw_bars()

    predicted = int(pred[0].argmax(0))
    # print(predicted)
    label_target = infoCanvas.find_withtag(TAG_NUM)
    infoCanvas.itemconfigure(label_target, text=f"{predicted}")

    return output


def reset_painted_cell():
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            grid_cells_value[(x, y)] = 0


def get_activations(model: NeuralNetwork, x: torch.Tensor):
    # 读取模型预测 x 后的，各个网络层的激活值，并保存到全局变量
    global activation_values
    activation_values = []
    for name, module in model.named_modules():
        # print(f"name: {name}, module: {module}")
        if isinstance(module, nn.Linear):  # 若模块是线性层（全连接层）
            x = module(x)
            activation_values.append(x.detach().numpy())  # 存储激活值
            x = torch.relu(x)  # 假设使用ReLU激活函数


def standardization(data: np.ndarray):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


# def normalization(data):
#     _range = np.max(abs(data))
#     return data / _range


def normalization(data: np.ndarray):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def get_activation_values(x: torch.Tensor):
    # 调用函数进行前向传播并收集激活值
    get_activations(model, x.reshape(-1))

    first_hidden_layer_weights = model.linear_relu_stack[0].weight.data
    first_hidden_layer_bias = model.linear_relu_stack[0].bias.data

    # 获取第二层（实际上是第二个线性层，因为ReLU不是参数层）的权重和偏置
    second_hidden_layer_weights = model.linear_relu_stack[2].weight.data
    second_hidden_layer_bias = model.linear_relu_stack[2].bias.data

    # 获取输出层的权重和偏置
    output_layer_weights = model.linear_relu_stack[4].weight.data
    output_layer_bias = model.linear_relu_stack[4].bias.data

    global olwn, slwn
    olw = output_layer_weights.numpy()
    olwn = normalization(olw * activation_values[1])

    slw = second_hidden_layer_weights.numpy()
    slwn = normalization(slw * activation_values[0])

    # # 输出隐藏层的激活值
    # for i, activation in enumerate(activation_values):
    #     print(f"第{i}层的激活值:")
    #     print(activation)


def redraw_bars():
    # 绘制每个神经元上的柱状图
    bars = networkCanvas.find_withtag(TAG_BAR)
    for bar in bars:
        networkCanvas.delete(bar)

    # 各个隐藏层的激活值，标准化处理一下到 [0, 1] 区间
    act1 = normalization(activation_values[2])
    act2 = normalization(activation_values[1])
    act3 = normalization(activation_values[0])
    # 绘制输出层的10个神经元
    for i in range(10):
        tag1 = f"n{i}"
        cell = networkCanvas.find_withtag(tag1)

        c1 = networkCanvas.coords(cell)  # 获取单元的坐标
        x1, y1 = (
            c1[0] + 3,
            c1[3] - 3,
        )
        x2, y2 = (
            c1[2] - 3,
            c1[3] - 3 - 44 * act1[i],
        )
        networkCanvas.create_rectangle(
            x1, y1, x2, y2, fill="#39B1EC", width=0, tag=TAG_BAR
        )
    # 绘制隐藏层的神经元，每层64个
    for i in range(64):
        tag1 = f"b{i}"
        tag2 = f"a{i}"
        # 第二隐藏层
        cell1 = networkCanvas.find_withtag(tag1)
        c1 = networkCanvas.coords(cell1)  # 获取单元的坐标
        x1, y1 = (
            c1[0] + 0,
            c1[3] - 0,
        )
        x2, y2 = (
            c1[2] - 0,
            c1[3] - 0 - 48 * act2[i],
        )
        networkCanvas.create_rectangle(
            x1, y1, x2, y2, fill="#39B1EC", tag=TAG_BAR, width=0
        )
        # 第一隐藏层
        cell2 = networkCanvas.find_withtag(tag2)
        c1 = networkCanvas.coords(cell2)  # 获取单元的坐标
        x1, y1 = (
            c1[0] + 0,
            c1[3] - 0,
        )
        x2, y2 = (
            c1[2] - 0,
            c1[3] - 0 - 48 * act3[i],
        )
        networkCanvas.create_rectangle(
            x1, y1, x2, y2, fill="#39B1EC", tag=TAG_BAR, width=0
        )


def redraw_lines():
    lines = networkCanvas.find_withtag(TAG_LINE)
    for line in lines:
        networkCanvas.delete(line)

    for i in range(10):
        tag1 = (f"n{i}",)
        neure1 = networkCanvas.find_withtag(tag1)
        c1 = networkCanvas.coords(neure1)  # 获取单元的坐标
        x1, y1 = c1[0] + 15, c1[3]
        for j in range(64):
            tag2 = (f"b{j}",)
            neure2 = networkCanvas.find_withtag(tag2)
            c2 = networkCanvas.coords(neure2)  # 获取单元的坐标
            x2, y2 = c2[0] + 2, c2[1]

            # 输出层和第二隐藏层之间的连线，连线太多，只显示激活值比较高的部分
            global olwn
            if olwn[i, j] > 0.55:
                networkCanvas.create_line(x1, y1, x2, y2, fill="#2D8EBC", tag=TAG_LINE)

    for j in range(64):
        tag2 = (f"b{j}",)
        neure2 = networkCanvas.find_withtag(tag2)
        c2 = networkCanvas.coords(neure2)  # 获取单元的坐标
        x2, y2 = c2[0] + 2, c2[3]
        for k in range(64):
            tag3 = (f"a{k}",)
            neure3 = networkCanvas.find_withtag(tag3)
            c3 = networkCanvas.coords(neure3)  # 获取单元的坐标
            x3, y3 = c3[0] + 2, c3[1]
            # 第一隐藏层和第二隐藏层之间的连线，太多，随机显示十分之一，且激活值比较高的
            if np.random.randint(10) == 1:
                global slwn
                if slwn[i, j] > 0.4:
                    networkCanvas.create_line(
                        x2, y2, x3, y3, fill="#2D8EBC", tag=TAG_LINE
                    )


#############################################################################
# 绘制主程序界面，用全局变量便于访问
root = tk.Tk()
root.resizable(width=False, height=False)
# 创建顶部的文本信息 Canvas
infoCanvas = tk.Canvas(
    root,
    height=220,
    width=600,
)  # 调整canvas宽度以适应窗口宽度并留出边距
infoCanvas.create_text(
    290,
    80,
    text="""
    这是一个有2个隐藏层的多层感知人工神经网络，每个隐
    藏层有64个神经元，演示了在手写数字识别的过程中，
    神经网络的神经元如何工作，神经元的权重、偏置如何
    参与计算过程、传递激活值，并最终预测手写数字。
    """,
    fill="#264B76",
    font=("Arial", 15),
    justify="center",
)
infoCanvas.create_text(
    300,
    180,
    text="X",
    tag=TAG_NUM,
    fill="#264B76",
    font=("Arial", 64),
    justify="center",
)
infoCanvas.pack()

# 创建 绘制神经网络 Canvas
networkCanvas = tk.Canvas(root, height=450, width=600)
for i in range(10):
    networkCanvas.create_text(
        65 + 50 * i,
        22,
        text=str(i),
        fill="#264B76",
        font=("Airal", 24),
        justify="center",
    )
    networkCanvas.create_rectangle(
        50 + 50 * i, 50, 50 + 50 * i + 30, 100, fill="white", width=0, tag=f"n{i}"
    )
for i in range(64):
    networkCanvas.create_rectangle(
        44 + 8 * i, 200, 44 + 8 * i + 5, 250, fill="white", width=0, tag=f"b{i}"
    )
    networkCanvas.create_rectangle(
        44 + 8 * i, 350, 44 + 8 * i + 5, 400, fill="white", width=0, tag=f"a{i}"
    )
networkCanvas.pack()

# 创建手写体输入 Canvas
writingCanvas = tk.Canvas(
    root, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE, bg="#EFEFEF"
)
for i in range(GRID_SIZE):
    writingCanvas.create_line(
        0, i * CELL_SIZE, CELL_SIZE * GRID_SIZE, i * CELL_SIZE, fill="white"
    )
    writingCanvas.create_line(
        i * CELL_SIZE, 0, i * CELL_SIZE, CELL_SIZE * GRID_SIZE, fill="white"
    )
writingCanvas.pack()


# 绑定鼠标按下与移动事件到 move_brush 函数
writingCanvas.bind("<B1-Motion>", move_brush)

# 创建一个无标题的ttk Frame来组织按钮
frame_buttons = ttk.Frame(root, padding=15)
frame_buttons.pack(
    side=tk.BOTTOM, fill=tk.X, pady=(10, 0)
)  # 增加底部pady以提供更多空间

# 创建重置按钮和调试按钮
reset_button = ttk.Button(frame_buttons, text="Reset", command=reset_canvas)
read_button = ttk.Button(frame_buttons, text="Debug", command=get_gray_value)

# 使用grid布局让按钮水平居中
reset_button.grid(row=0, column=0, padx=(0, 5), pady=5, sticky=tk.W + tk.E)
read_button.grid(row=0, column=1, padx=(5, 0), pady=5, sticky=tk.W + tk.E)

# 因为frame_buttons填充了整个水平方向，所以按钮会居中
frame_buttons.columnconfigure(0, weight=1)
frame_buttons.columnconfigure(1, weight=1)

reset_canvas()
root.mainloop()
