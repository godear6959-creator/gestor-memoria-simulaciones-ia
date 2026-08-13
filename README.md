# # Gestor de Memoria para Simulaciones e IA

Utilidades en Python para procesar modelos, datasets, imágenes y archivos grandes sin saturar RAM/VRAM — pensado para correr en hardware limitado (GPUs modestas, equipos salvados).

## Problema que resuelve

En simulaciones de IA (entrenamiento, inferencia, datasets grandes) es común toparse con `CUDA out of memory` o `MemoryError` porque:
- El modelo se carga completo en float32.
- El dataset se carga entero en RAM.
- Los tensores intermedios no se liberan entre iteraciones.

## Solución

1. **Achicar** — cuantización dinámica (int8) para capas Linear/LSTM, y conversión a float16 para el resto del modelo.
2. **Partir** — procesamiento de datasets `.npy`/`.csv` en bloques (`chunksize` / `mmap_mode="r"`), sin cargar el archivo completo.
3. **Limpiar** — `torch.cuda.empty_cache()` + `gc.collect()` después de cada bloque procesado.

También incluye compresión de imágenes por lote (OpenCV) y compresión de archivos grandes en streaming (Zstandard).

## Uso

```bash
pip install torch numpy pandas zstandard opencv-python-headless
python gestion_memoria_simulaciones_ia.py
```

Menú interactivo:
1. Optimizar/cuantizar modelo PyTorch (.pt)
2. Procesar dataset por bloques (.npy o .csv)
3. Comprimir lote de imágenes
4. Comprimir archivo con Zstandard

## Nota sobre cuantización

`torch.quantization.quantize_dynamic` solo cuantiza capas `Linear` y `LSTM`. Las capas `Conv2d` **no se reducen** con este método — si tu modelo es mayormente convolucional, usá la salida `_fp16.pt` (afecta todas las capas) o implementá cuantización estática con calibración aparte.

## Licencia

MITgestor-memoria-simulaciones-ia
