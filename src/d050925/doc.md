# Linear Models and Learning via Optimization

- Tìm w1, w2, ...., wd để dự đoán y

$$
W = \begin{bmatrix}
    w1 \\
    w2 \\
    ... \\
    wd
\end{bmatrix}
vector trọng số
$$

- Để vector lượng W. Sử dụng dữ liệu: training data
$$N: \{(x1, y1), (x2, y2), ..., (xn, yn)\}$$

$$
xn = \begin{bmatrix}
    x11 & x12 & ... & x1d \\
    x21 & x22 & ... & x2d \\
    ... & ... & ... & ... \\
    xn1 & xn2 & ... & xnd
\end{bmatrix}
vector đặc trưng
$$
- Sau đó có W, ta có thể dự đoán y
$$
y = f(W^T * x)= W^T * x
$$
- Không bao giờ tìm được W1, W2, ..., Wd sao cho:
$$
y1 = W1 * x1^{1} + W2 * x2^{1} + ... + Wd * xd^{1} \\
y2 = W1 * x1^{2} + W2 * x2^{2} + ... + Wd * xd^{2} \\
... \\
yn = W1 * x1^{n} + W2 * x2^{n}
$$
Mà chỉ tìm được gần đúng. Vậy ta cần hàm lỗi (loss function) để đánh giá độ sai lệch giữa giá trị dự đoán và giá trị thực tế.
- Hàm lỗi (loss function):
$$
L(y, W^{T}x_n)
$$
Là hàm mất mát dữ liệu tại điểm thứ n

- Squared loss function:
Hàm khả vi dễ tính đạo hàm
$$
L(y_n, f(x_n)) = (y_n - W^{T}x_n)^2 
$$
- Absolute loss function: Tính đạo hàm khó, 1 số điều không tồn tại đạo hàm
$$
L(y_n, f(x_n)) = |y_n - f(x_n)|
$$
- Huber loss:
$$
L(y_n, f(x_n)) = 
\begin{cases}
    |y_n - f(x_n)| & \text{nếu } |y_n - f(x_n)| > \delta \\
    (y_n - f(x_n))^2 & \text{nếu } |y_n - f(x_n)| \leq \delta
\end{cases}
$$
- Insensitive loss:
$$
L(y_n, f(x_n)) = 
\begin{cases}
    0 & \text{nếu } |y_n - f(x_n)| \leq \delta \\
    |y_n - f(x_n)| - \delta & \text{nếu } |y_n - f(x_n)| > \delta
\end{cases}
$$

- Mục tiêu: cực tiểu hàm loss
  + Giải hệ phương trình:
  $$
  \frac{\partial L(w)}{\partial w} = 0 \text{ với } L(w) = \sum_{n=1}^{N} L(y_n, f(x_n))
  $$
  + Gradient Descent:
    $$
    w^{(t+1)} := w^{(t)} - \eta \frac{\partial L(w^{(t)})}{\partial w}(w^{(t)})
    $$

## Ví dụ 1:
- Dữ liệu:
  | x1 | x2 | y  |
  |----|----|----|
  | 1  | 2  | 1  |
  | 2  | 3  | 3  |
  | 4  | 2  | 3  |
  | 5  | 3  | 4  |
  | 2  | 4  | 3  |
- Hàm f:
$$
f(x_n) = W^{T}x_n = w1 * x1 + w2 * x2
$$
- Hàm loss:
$$
l(y_{n}, f(x_{n})) = (y_n - f(x_n))^2
$$

$$
L(W) = \sum_{n=1}^{N} l(y_n, f(x_n)) 
$$
### Giải
$$
L(w) = \sum_{n=1}^{5} (y_n - W_{1}x_{1}^{1} - W_{2}x_{2}^{1})^2
$$
$$
= (2 - W_{1} - W_{2}*2)^2 + (3 - W_{1}*2 - W_{2}*3)^2 + (3 - W_{1}*4 - W_{2}*2)^2 + (4 - W_{1}*5 - W_{2}*3)^2 + (3 - W_{1}*2 - W_{2}*4)^2
$$
$$
= 5W_{1}^{2} + 42W_{2}^{2} + 34W_{1}W_{2} - 52W_{1} - 92W_{2} + 48
$$

- Tính hệ:
$$
\frac{\partial L(w)}{\partial w} = 0
$$
$$
\begin{bmatrix}
    \frac{\partial L(w)}{\partial W_{1}} = 0 \\ \\
    \frac{\partial L(w)}{\partial W_{2}} = 0
\end{bmatrix}
$$
$$
\begin{bmatrix}
    50W_1 + 39W_2 -46 = 0
    \\ \\
    39W_1 + 42W_2 - 43 = 0
\end{bmatrix}
$$

- Gradient descent:
$$
W^{0}_{1} = 1; W^{0}_{2} = 2; \eta = 0.01
$$

$$
W^{(t+1)} := W^{(t)} - \eta \frac{\partial L(W^{(t)})}{\partial W}(W^{(t)})
$$

$$
\frac{\partial L(W)}{\partial W} =
\begin{bmatrix}
    \frac{\partial L(W)}{\partial W_{1}} \\ \\
    \frac{\partial L(W)}{\partial W_{2}}
\end{bmatrix} =
\begin{bmatrix}
    10W_{1} + 34W_{2} - 52 \\ \\
    34W_{1} + 84W_{2} - 92
\end{bmatrix}
$$

- Vòng lặp:
  + t = 0:
$$
W^{1} = W^{0} - 0.01 *
\begin{bmatrix}
    10*1 + 34*2 - 52 \\ \\
    34*1 + 84*2 - 92
\end{bmatrix} =
\begin{bmatrix}
    1 \\ \\
    2
\end{bmatrix} - 0.01 *
\begin{bmatrix}
    6 \\ \\
    110
\end{bmatrix} =
\begin{bmatrix}
    0.94 \\ \\
    0.9
\end{bmatrix}
$$
  + t = 1:
$$
W^{2} = W^{1} - 0.01 * \begin{bmatrix}
    10*0.94 + 34*0.9 - 52 \\
    34*0.94 + 84*0.9 - 92
\end{bmatrix}
$$ 
$$=
\begin{bmatrix}
    0.94 \\ \\
    0.9
\end{bmatrix} - 0.01 *
\begin{bmatrix}
    -20.4 \\ \\
    1.96
\end{bmatrix} =
\begin{bmatrix}
    1.144 \\ \\
    0.8804
\end{bmatrix}
$$
  + t = ...: (tương tự)