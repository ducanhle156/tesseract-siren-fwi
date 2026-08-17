# SIREN for FWI

## Nguồn

Mạng SIREN được lấy từ
[KronosAI_solutions/Task1](https://github.com/nguyenvanhaibk92/KronosAI_solutions/tree/main/Task1).
Repo đó là một **PINN cho bài toán điện từ** (Maxwell + PML + level-set
permittivity); SIREN chỉ là mạng nền bên trong. Chỉ phần SIREN được clone về.

### Lấy về

| File gốc (Task1/utils) | Ở đây | Thay đổi |
|---|---|---|
| `siren_network.py` | `siren_network.py` | `omega_0` truyền vào thay vì đọc từ `CONFIG` toàn cục |
| — | `_siren_network_original.py` | bản gốc nguyên vẹn, để đối chiếu |
| recipe optax trong `training.py` | `velocity_field.fit()` | giữ warmup-cosine + clip global norm |

### Không lấy

`pinn_model.py`, `physics_functions.py` (Maxwell/PML), `permitivity_level_set.py`,
`point_generation.py`, `phase2_training.py`, `plots.py`, `config.yaml`,
`main.py`, các `models/*.pkl` — tất cả đều gắn với bài toán điện từ, không dùng
cho FWI.

## Cách hoạt động

Thay vì nghịch đảo trực tiếp 601×221 = 132.821 giá trị lưới, vận tốc được biểu
diễn bằng một mạng tọa độ:

```
vp(x, z) = vmin + (vmax - vmin) * sigmoid(SIREN([x_norm, z_norm]))
```

Ẩn số của bài toán trở thành trọng số mạng. Devito vẫn trả về `dL/dvp` trên
lưới; một VJP qua mạng chuyển nó thành `dL/dtheta`:

```
dL/dtheta = (dvp/dtheta)^T @ (dL/dvp)
```

Devito hoàn toàn không biết có mạng — nó vẫn nhận vào một mảng `(nx, nz)` và
trả ra gradient `(nx, nz)`.

`sigmoid` ràng buộc vận tốc trong `[vmin, vmax]` nên L-BFGS-B không cần bound.

## Dùng

```python
from siren import SirenVelocity

siren = SirenVelocity(shape=(601, 221), spacing=(0.015, 0.015),
                      vmin=1.5, vmax=4.7, hidden=(256,)*5, omega_0=30.)

siren.fit(vp_init, n_epochs=3000)      # warm start: khớp mạng với model ban đầu
vp = siren.vp()                        # (nx, nz) float32 -> đưa vào Devito
loss, grad_vp = pool.loss_grad_vp(vp)  # FWI trả gradient trên lưới
grad_theta = siren.chain_grad(grad_vp) # chain rule -> gradient trọng số
```

Chạy FWI đầy đủ:

```bash
python AcousticVel_L2_SIREN.py --nprocs 20 --threads 1
python AcousticVel_L2_SIREN.py --benchmark          # đo 1 gradient rồi thoát
```

## Tham số đáng chỉnh

- `--omega0` (mặc định 30): tần số của sine. Cao → mạng biểu diễn được chi tiết
  mịn hơn nhưng dễ sinh nhiễu; thấp → model mượt hơn. Đây là "núm" regularisation
  chính.
- `--hidden` / `--layers`: dung lượng mạng. Mặc định 256×5 ≈ 264k trọng số, tức
  ~2× số ẩn lưới — chưa nén chiều. Muốn regularise mạnh hơn thì giảm xuống,
  ví dụ `--hidden 128 --layers 4` (~50k).
- `--fit-epochs`: pre-fit vào model ban đầu. Bỏ qua bước này thì FWI xuất phát
  từ một trường vận tốc ngẫu nhiên vô nghĩa.

## Đã kiểm chứng

- `chain_grad` khớp autodiff với sai số tương đối **0.0**, và khớp finite
  difference trên các trọng số ngẫu nhiên.
- Chạy end-to-end trong env `geo_jxli` (2×32-core Xeon 6448Y, `--nprocs 20
  --threads 1`):

  | | loss+gradient / 20 shot | loss xuất phát |
  |---|---|---|
  | bản lưới (`AcousticVel_L2_1stage_parallel.py`) | 6.3 s | 4.18e-07 |
  | bản SIREN (`--fit-epochs 300`) | 7.6 s | 2.11e-06 |

  Loss xuất phát của bản SIREN cao hơn ~5× vì mạng chưa fit khít model ban đầu;
  tăng `--fit-epochs` sẽ kéo về gần bản lưới. Chênh 1.3 s mỗi gradient là chi phí
  forward + VJP của mạng — nhỏ so với lan truyền sóng.

- Chất lượng pre-fit theo số epochs (mạng 256×5, JAX trên CPU, model ban đầu
  Marmousi đã mask lớp nước):

  | `--fit-epochs` | thời gian | RMSE so với model ban đầu |
  |---|---|---|
  | 300 | 122 s | 0.0497 km/s |
  | 1500 | 524 s | 0.0146 km/s |
  | 4000 | 1344 s | 0.0052 km/s |

  RMSE giảm gần tuyến tính theo log(epochs) và chưa bão hòa ở 4000. Mặc định
  3000 là điểm cân bằng (~17 phút, RMSE ~0.007 km/s ≈ 0.2% dải vận tốc) — đủ
  khít để FWI xuất phát từ đúng model ban đầu. Bước fit này chạy một lần trước
  khi nghịch đảo, không lặp lại mỗi iteration.

## Lưu ý

- JAX chạy CPU (`JAX_PLATFORMS=cpu` đặt trong script). Máy có GPU nhưng jaxlib
  cài là bản CPU-only; mạng nhỏ nên không phải nút thắt so với Devito.
- Water-layer mask vẫn áp trong `PostProcessVP` như bản lưới, nhưng ở đây mask
  chỉ triệt tiêu gradient chứ không khóa cứng vận tốc lớp nước — mạng vẫn có thể
  thay đổi vùng đó gián tiếp qua trọng số dùng chung.
