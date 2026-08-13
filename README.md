# 🧠 Gestión de Memoria en Simulaciones de IA

> **No hay que meter todo el problema de una vez en la memoria — hay que partirlo en pedazos manejables, achicar lo que se pueda, y botar lo que ya no sirve antes de seguir.**

---

## 🛑 El problema

Metiste más de lo que cabe, y no botaste lo que ya no sirve.

En simulaciones de IA (entrenamiento, inferencia, datasets grandes), la RAM o la GPU se llenan porque:
- El modelo completo se carga en memoria a máxima precisión (`float32`).
- El dataset se carga entero, en vez de a pedazos.
- Los tensores intermedios que ya no se usan quedan reservados en la memoria de la GPU.

Resultado: `CUDA out of memory`, `MemoryError`, o el proceso muere silenciosamente (OOM killer).

---

## 💡 La solución

1. **Achica** el modelo — usa `int8`/`float16`, no `float32`.
2. **Parte** el dataset en pedazos (chunks), no lo cargues entero.
3. **Limpia** la basura después de cada pedazo (`empty_cache()` + `gc.collect()`).

No metas todo junto. No dejes la cáscara.

---

## 📦 Instalación

```bash
pip install torch numpy pandas zstandard opencv-python-headless
```

Opcionales pero recomendados:
```bash
pip install psutil  # para monitoreo de RAM en tiempo real
```

---

## 🚀 Uso rápido

### 1. Optimizar/Cuantizar un modelo (achicar)

```python
from gestion_memoria_simulaciones_ia import optimizar_modelo_ia

# Guarda AMBAS versiones: int8 (capas Linear) y fp16 (todas las capas)
# Detecta automáticamente si hay capas Conv2d y te avisa
optimizar_modelo_ia("modelo_original.pt", "modelo_optimizado.pt")

# Salida:
# ✅ modelo_optimizado.pt        (int8, Linear)
# ✅ modelo_optimizado_fp16.pt   (float16, TODAS las capas)
```

> ⚠️ **Nota sobre Conv2d**: `torch.quantization.quantize_dynamic` solo reduce capas `Linear` y `LSTM`. Si tu modelo tiene `Conv2d`, el int8 no las comprimirá. Usá la versión `_fp16.pt` si necesitas reducir TODO el modelo.

### 2. Procesar dataset por bloques (partir)

```python
from gestion_memoria_simulaciones_ia import procesar_dataset_chunks

# NPY: memory mapping (lee del disco, no de RAM)
procesar_dataset_chunks(
    ruta_entrada="embeddings_50gb.npy",
    ruta_salida_prefijo="./resultados/embeddings",
    chunk_size=5000
)
# Salida: ./resultados/embeddings_chunk_0.npy, embeddings_chunk_5000.npy, ...

# CSV: streaming de a 10,000 filas
procesar_dataset_chunks(
    ruta_entrada="dataset_20gb.csv",
    ruta_salida_prefijo="./resultados/dataset",
    chunk_size=10000,
    procesador=lambda df: df.dropna()  # tu lógica de procesamiento
)
# Salida: ./resultados/dataset_chunk_0.csv, dataset_chunk_1.csv, ...
```

### 3. Procesar tensor en GPU por batches (no todo junto)

```python
from gestion_memoria_simulaciones_ia import procesar_tensor_en_gpu

# Solo un batch en VRAM a la vez
resultado = procesar_tensor_en_gpu(
    tensor_cpu=mi_tensor_grande,
    modelo=mi_modelo,
    batch_size=32,
    dispositivo="cuda"
)
```

### 4. Comprimir imágenes una por una

```python
from gestion_memoria_simulaciones_ia import comprimir_imagenes_directorio

comprimir_imagenes_directorio(
    dir_entrada="./imagenes_crudas/",
    dir_salida="./imagenes_comprimidas/",
    calidad=85,
    max_dimension=1024  # redimensiona si excede este tamaño
)
```

### 5. Comprimir archivos grandes por streaming

```python
from gestion_memoria_simulaciones_ia import comprimir_zstd, descomprimir_zstd

# Zstandard con copy_stream (máxima eficiencia, sin RAM)
comprimir_zstd("logs_100gb.txt", "logs_100gb.txt.zst", nivel=3)

# Descomprimir
descomprimir_zstd("logs_100gb.txt.zst", "logs_recuperados.txt")
```

> Si `zstandard` no está instalado, usa gzip automáticamente como fallback.

---

## 🔧 La función clave: `limpiar_memoria()`

```python
def limpiar_memoria():
    if torch.cuda.is_available():
        torch.cuda.synchronize()   # Espera que la GPU termine
        torch.cuda.empty_cache()     # Libera caché reservada pero no usada
    gc.collect()                      # Limpia referencias muertas de Python
```

**Se llama después de cada bloque, batch o imagen procesada.**
Es lo que evita que la memoria se llene de a poco hasta reventar.

---

## ✅ Checklist para tus simulaciones

- [ ] ¿Cargás el dataset de entrenamiento completo en RAM? → Usá `mmap_mode="r"` o `chunksize`.
- [ ] ¿Tenés tensores intermedios en GPU que no volvés a usar? → `del tensor` + `torch.cuda.empty_cache()`.
- [ ] ¿El modelo pesa más de lo que tolera tu hardware? → Cuantización dinámica o `.half()`.
- [ ] ¿Tu modelo tiene capas Conv2d? → La versión int8 NO las reduce; usá la versión fp16.
- [ ] ¿Procesás logs de inferencia o series temporales largas? → Streaming con zstandard.
- [ ] ¿Llamás `limpiar_memoria()` después de **cada** iteración del loop principal?

---

## 📊 ¿Por qué funciona?

| Técnica | Memoria antes | Memoria después | Precisión |
|---------|--------------|-----------------|-----------|
| Cuantización int8 | float32 (4 bytes/número) | int8 (1 byte/número) | ~4× menos memoria |
| Cuantización fp16 | float32 (4 bytes/número) | float16 (2 bytes/número) | ~2× menos memoria |
| Dataset por chunks | Todo el archivo en RAM | Solo el chunk activo | Sin pérdida |
| GPU por batches | Tensor completo en VRAM | Solo el batch activo | Sin pérdida |
| Streaming zstd | Archivo completo en RAM | Buffer de 1 MB | Sin pérdida |

---

## 🏗️ Framework

Este script forma parte del ecosistema **PNLIO** (Kernel v1.1, Innova v6/v7), diseñado para correr 100% local/offline con **Ollama + Phi-3 + ChromaDB**, optimizado para hardware limitado (hasta 5 GB RAM).

- 📖 Autor: **Gonzalo Mauricio de la Rivera Arellano**
- 🔗 GitHub: [@godear6959-creator](https://github.com/godear6959-creator)
- 🎨 ArtStation: [gonzalodelarivera8](https://www.artstation.com/gonzalodelarivera8)
- 📦 Gumroad: [godear](https://gumroad.com/godear)
- 🆔 ORCID: [0009-0001-9455-8416](https://orcid.org/0009-0001-9455-8416)
- 📚 Autor de *"La Sinfonía de la Realidad"* (2025)

---

## 📄 Licencia

MIT — Usalo, modificálo, compartilo. Si te salvó de un OOM a las 3 AM, contá la historia.

---

> *"No hay que meter todo el problema de una vez en la memoria."*
