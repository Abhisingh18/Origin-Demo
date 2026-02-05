# MANTIS – Text Conditioned Segmentation

This project implements a text-conditioned image segmentation system for:

- Crack segmentation
- Drywall taping / joint area segmentation

The system accepts an image and a natural language prompt and returns a binary segmentation mask.

---

## � Demo

- **Frontend (Vercel):** https://YOUR-FRONTEND.vercel.app
- **Backend (Render):** https://YOUR-BACKEND.onrender.com

---

## 🧠 Approach

- Single segmentation model (ResNet18 + UNet decoder)
- Prompt → class mapping
- Same image + different prompt → different segmentation mask

This design ensures:
- Correctness
- Consistency
- Simple deployment

---

## 📊 Datasets

| Dataset | Task | Split |
|------|------|------|
| Drywall-Join-Detect | Taping area | Train / Val |
| Cracks | Crack detection | Train / Val |

---

## 📈 Metrics

- Dice Score
- mIoU

Training curves and qualitative results are provided in `/reports`.

---

## �️ Inference

```bash
POST /predict
- image
- prompt
```

Returns:
- **Format**: PNG
- **Type**: Single channel (Binary Mask)
- **Values**: {0, 255}

## ⚙️ Deployment

- **Backend**: FastAPI + Render
- **Frontend**: Static/Next.js + Vercel

## 🏁 Results

- Stable across varied scenes
- Handles thin cracks and joint areas
- Fast inference (~400 ms / image on CPU)

## 📌 Notes

RTX 5070 GPU is not yet supported by PyTorch; training and inference are performed on CPU.

## 📜 License

MIT
