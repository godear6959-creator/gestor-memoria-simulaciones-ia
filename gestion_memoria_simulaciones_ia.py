#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gestion_memoria_simulaciones_ia.py
==================================
Gestión de memoria en simulaciones de IA: no metas todo junto, no dejes la cáscara.

Técnicas:
  1. Cuantización de modelos (int8 / float16) con detección de capas no soportadas
  2. Datasets por bloques (mmap + chunksize) con callback de procesamiento
  3. Tensores en GPU por batches (solo un batch en VRAM a la vez)
  4. Imágenes una por una (OpenCV headless)
  5. Archivos grandes por streaming (Zstandard / gzip fallback)
  6. Limpieza proactiva de memoria (CUDA + GC)

Autor: Gonzalo Mauricio de la Rivera Arellano
Framework: PNLIO (Kernel v1.1, Innova v6/v7)
"""

import os
import gc
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Callable

# ------------------------------------------------------------------
# Dependencias opcionales con fallback
# ------------------------------------------------------------------
try:
    import zstandard as zstd
    ZSTANDARD_DISPONIBLE = True
except ImportError:
    ZSTANDARD_DISPONIBLE = False
    print("⚠️  zstandard no instalado. Compresión usará gzip como fallback.")

try:
    import cv2
    CV2_DISPONIBLE = True
except ImportError:
    CV2_DISPONIBLE = False
    print("⚠️  opencv-python no instalado. Funciones de imagen no disponibles.")


# ============================================================
# 0. LIMPIEZA DE MEMORIA
# ============================================================

def limpiar_memoria():
    """
    Libera caché de CUDA y fuerza garbage collection.
    Se llama después de cada bloque/batch procesado.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()   # Espera que la GPU termine antes de liberar
        torch.cuda.empty_cache()     # Libera memoria reservada pero no ocupada
    gc.collect()


# ============================================================
# 1. MODELOS DE IA (CUANTIZACIÓN)
# ============================================================

def optimizar_modelo_ia(ruta_modelo: Union[str, Path], ruta_salida: Union[str, Path]):
    """
    Reduce un modelo grande (float32) a precisión int8 o float16.
    Guarda AMBAS versiones: int8 (capas Linear) y fp16 (todas las capas).

    Args:
        ruta_modelo: Ruta al modelo .pt original.
        ruta_salida: Ruta base para guardar los modelos optimizados.
    """
    print("\n--- [1] Optimización de Modelo de IA ---")
    ruta_modelo = Path(ruta_modelo)
    ruta_salida = Path(ruta_salida)

    if not ruta_modelo.exists():
        print(f"❌ Error: El archivo {ruta_modelo} no existe.")
        return

    # Cargar modelo a CPU (evita OOM en GPU durante la carga)
    model = torch.load(ruta_modelo, map_location="cpu")
    print(f"📦 Modelo cargado en CPU: {ruta_modelo.name}")

    # Detectar capas Conv2d (quantize_dynamic NO las reduce)
    tiene_conv = any(isinstance(m, torch.nn.Conv2d) for m in model.modules())
    if tiene_conv:
        print("⚠️  [AVISO] El modelo tiene capas Conv2d. La cuantización dinámica "
              "NO las reduce (solo afecta Linear/LSTM). Para comprimir Conv2d se "
              "necesita cuantización estática con calibración.")

    # --- Versión int8 (solo Linear) ---
    model_quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    ruta_int8 = ruta_salida.with_suffix(".pt") if ruta_salida.suffix != ".pt" else ruta_salida
    torch.save(model_quantized, ruta_int8)
    tamano_int8 = ruta_int8.stat().st_size / 1e6
    print(f"✅ Modelo cuantizado (int8, capas Linear) → {ruta_int8.name} ({tamano_int8:.1f} MB)")

    # --- Versión float16 (TODAS las capas, incluida Conv2d) ---
    ruta_fp16 = ruta_int8.with_stem(ruta_int8.stem + "_fp16")
    model_half = model.half()
    torch.save(model_half, ruta_fp16)
    tamano_fp16 = ruta_fp16.stat().st_size / 1e6
    print(f"✅ Modelo comprimido (float16, todas las capas) → {ruta_fp16.name} ({tamano_fp16:.1f} MB)")

    # Liberar
    del model, model_quantized, model_half
    limpiar_memoria()
    print("🧹 Memoria liberada.\n")


# ============================================================
# 2. DATASETS POR BLOQUES
# ============================================================

def procesar_dataset_chunks(
    ruta_entrada: Union[str, Path],
    ruta_salida_prefijo: Union[str, Path],
    chunk_size: int = 10000,
    procesador: Optional[Callable] = None
):
    """
    Lee datasets gigantes (.npy / .csv) en bloques pequeños de memoria.

    Args:
        ruta_entrada: Archivo .npy o .csv a procesar.
        ruta_salida_prefijo: Prefijo para los archivos resultantes.
        chunk_size: Tamaño de cada bloque.
        procesador: Función opcional para transformar cada chunk.
    """
    print(f"\n--- [2] Procesamiento por Bloques (chunk_size={chunk_size}) ---")
    ruta_entrada = Path(ruta_entrada)
    ruta_salida_prefijo = Path(ruta_salida_prefijo)

    if not ruta_entrada.exists():
        print(f"❌ Error: El archivo {ruta_entrada} no existe.")
        return

    extension = ruta_entrada.suffix.lower()
    total_procesados = 0

    if extension == ".npy":
        data = np.load(ruta_entrada, mmap_mode="r")
        n_total = data.shape[0]
        print(f"📦 Dataset NPY mapeado en disco: {n_total:,} filas")

        for i in range(0, n_total, chunk_size):
            chunk = data[i:i + chunk_size].copy()

            if procesador:
                chunk = procesador(chunk)
            else:
                chunk = chunk * 2  # operación de ejemplo

            ruta_out = ruta_salida_prefijo.parent / f"{ruta_salida_prefijo.name}_chunk_{i}.npy"
            np.save(ruta_out, chunk)
            total_procesados += len(chunk)

            del chunk
            limpiar_memoria()

            if (i // chunk_size + 1) % 10 == 0:
                print(f"    Bloques: {i // chunk_size + 1} | Filas: {total_procesados:,}")

        del data

    elif extension == ".csv":
        print(f"📦 Dataset CSV en streaming...")

        for i, chunk in enumerate(pd.read_csv(ruta_entrada, chunksize=chunk_size)):
            if procesador:
                chunk = procesador(chunk)
            else:
                chunk = chunk.dropna()

            ruta_out = ruta_salida_prefijo.parent / f"{ruta_salida_prefijo.name}_chunk_{i}.csv"
            chunk.to_csv(ruta_out, index=False)
            total_procesados += len(chunk)

            del chunk
            limpiar_memoria()

            if (i + 1) % 10 == 0:
                print(f"    Bloques: {i + 1} | Filas: {total_procesados:,}")

    else:
        print(f"❌ Formato no soportado: {extension}. Usa .csv o .npy")
        return

    limpiar_memoria()
    print(f"✅ Dataset procesado: {total_procesados:,} filas en bloques de {chunk_size}.\n")


# ============================================================
# 3. TENSORES EN GPU POR BATCHES
# ============================================================

def procesar_tensor_en_gpu(
    tensor_cpu: torch.Tensor,
    modelo: torch.nn.Module,
    batch_size: int = 32,
    dispositivo: str = "cuda"
) -> torch.Tensor:
    """
    Procesa un tensor grande moviendo solo batches a GPU, no todo el tensor.

    Args:
        tensor_cpu: Tensor completo en CPU.
        modelo: Modelo PyTorch (debe estar en el dispositivo objetivo).
        batch_size: Tamaño del batch que se mueve a GPU.
        dispositivo: "cuda" o "cpu".

    Returns:
        Tensor de resultados en CPU.
    """
    if dispositivo == "cuda" and not torch.cuda.is_available():
        dispositivo = "cpu"
        print("⚠️  CUDA no disponible, usando CPU")

    modelo.to(dispositivo)
    modelo.eval()

    n_total = len(tensor_cpu)
    resultados = []

    print(f"\n--- [3] Tensor en GPU por Batches (batch_size={batch_size}) ---")
    print(f"📦 Tensor: {n_total:,} elementos | Dispositivo: {dispositivo}")

    with torch.no_grad():
        for i in range(0, n_total, batch_size):
            batch = tensor_cpu[i:i + batch_size].to(dispositivo)
            salida = modelo(batch)
            resultados.append(salida.cpu())

            del batch, salida
            limpiar_memoria()

            if (i // batch_size + 1) % 10 == 0:
                print(f"    Batches: {i // batch_size + 1}/{n_total // batch_size + 1}")

    tensor_final = torch.cat(resultados, dim=0)
    del resultados
    limpiar_memoria()

    print(f"✅ Tensor procesado. Forma final: {tensor_final.shape}\n")
    return tensor_final


# ============================================================
# 4. IMÁGENES UNA POR UNA
# ============================================================

def comprimir_imagenes_directorio(
    dir_entrada: Union[str, Path],
    dir_salida: Union[str, Path],
    calidad: int = 85,
    max_dimension: Optional[int] = None
):
    """
    Comprime imágenes de un directorio una por una.

    Args:
        dir_entrada: Carpeta con imágenes.
        dir_salida: Carpeta destino.
        calidad: Calidad JPEG (1-100).
        max_dimension: Si se especifica, redimensiona manteniendo aspecto.

    Returns:
        Número de imágenes procesadas.
    """
    if not CV2_DISPONIBLE:
        raise ImportError("opencv-python no está instalado. Ejecuta: pip install opencv-python-headless")

    print(f"\n--- [4] Compresión de Imágenes (calidad={calidad}%) ---")
    dir_entrada = Path(dir_entrada)
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    extensiones = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    archivos = [f for f in dir_entrada.iterdir() if f.suffix.lower() in extensiones]

    print(f"📦 {len(archivos)} imágenes encontradas en {dir_entrada}")

    procesadas = 0
    for archivo in archivos:
        try:
            img = cv2.imread(str(archivo))
            if img is None:
                print(f"    ⚠️  No se pudo leer: {archivo.name}")
                continue

            if max_dimension:
                h, w = img.shape[:2]
                if max(h, w) > max_dimension:
                    escala = max_dimension / max(h, w)
                    img = cv2.resize(img, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA)

            path_out = dir_salida / f"{archivo.stem}.jpg"
            cv2.imwrite(str(path_out), img, [int(cv2.IMWRITE_JPEG_QUALITY), calidad])
            procesadas += 1
            del img
            limpiar_memoria()

        except Exception as e:
            print(f"    ❌ Error {archivo.name}: {e}")

    print(f"✅ {procesadas}/{len(archivos)} imágenes comprimidas → {dir_salida}\n")
    return procesadas


# ============================================================
# 5. ARCHIVOS GRANDES POR STREAMING
# ============================================================

def comprimir_zstd(
    ruta_entrada: Union[str, Path],
    ruta_salida: Union[str, Path],
    nivel: int = 3
):
    """
    Comprime un archivo grande usando Zstandard (o gzip fallback).
    Usa copy_stream para máxima eficiencia sin cargar en RAM.
    """
    print(f"\n--- [5] Compresión de Archivo ---")
    ruta_entrada = Path(ruta_entrada)
    ruta_salida = Path(ruta_salida)

    if not ruta_entrada.exists():
        print(f"❌ Error: El archivo {ruta_entrada} no existe.")
        return

    print(f"📦 Comprimiendo {ruta_entrada.name}...")

    if ZSTANDARD_DISPONIBLE:
        cctx = zstd.ZstdCompressor(level=nivel)
        with open(ruta_entrada, "rb") as f_in, open(ruta_salida, "wb") as f_out:
            cctx.copy_stream(f_in, f_out)
    else:
        import gzip
        with open(ruta_entrada, "rb") as f_in, gzip.open(ruta_salida, "wb", compresslevel=nivel) as f_out:
            while True:
                chunk = f_in.read(1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)

    tamano_orig = ruta_entrada.stat().st_size / 1e6
    tamano_comp = ruta_salida.stat().st_size / 1e6
    ratio = tamano_orig / tamano_comp if tamano_comp else 0

    print(f"✅ {tamano_orig:.1f} MB → {tamano_comp:.1f} MB (ratio: {ratio:.2f}x)")
    limpiar_memoria()


def descomprimir_zstd(
    ruta_entrada: Union[str, Path],
    ruta_salida: Union[str, Path]
):
    """Descomprime un archivo .zst o .gz por streaming."""
    print(f"\n--- [5b] Descompresión de Archivo ---")
    ruta_entrada = Path(ruta_entrada)
    ruta_salida = Path(ruta_salida)

    print(f"📦 Descomprimiendo {ruta_entrada.name}...")

    if ruta_entrada.suffix == ".zst" and ZSTANDARD_DISPONIBLE:
        dctx = zstd.ZstdDecompressor()
        with open(ruta_entrada, "rb") as f_in, open(ruta_salida, "wb") as f_out:
            dctx.copy_stream(f_in, f_out)
    else:
        import gzip
        with gzip.open(ruta_entrada, "rb") as f_in, open(ruta_salida, "wb") as f_out:
            while True:
                chunk = f_in.read(1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)

    tamano = ruta_salida.stat().st_size / 1e6
    print(f"✅ Descomprimido: {tamano:.1f} MB → {ruta_salida}\n")


# ============================================================
# 6. MENÚ INTERACTIVO
# ============================================================

def menu():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     🧠 GESTOR DE MEMORIA PARA SIMULACIONES E IA          ║
    ║     PNLIO / NIK Kernel Neuromórfico                      ║
    ╚══════════════════════════════════════════════════════════╝
    1. Optimizar/Cuantizar Modelo PyTorch (.pt)
    2. Procesar Dataset por Bloques (.npy o .csv)
    3. Procesar Tensor en GPU por Batches
    4. Comprimir Lote de Imágenes (OpenCV Headless)
    5. Comprimir Archivo con Zstandard
    6. Descomprimir Archivo
    7. Ver memoria disponible
    8. Salir
    """)

    while True:
        opcion = input("Selecciona una opción (1-8): ").strip()

        if opcion == "1":
            m_in = input("Ruta del modelo de entrada (.pt): ").strip().strip('"')
            m_out = input("Ruta de salida (.pt): ").strip().strip('"')
            optimizar_modelo_ia(m_in, m_out)

        elif opcion == "2":
            d_in = input("Ruta del dataset (.npy o .csv): ").strip().strip('"')
            d_out = input("Prefijo para archivos resultantes: ").strip().strip('"')
            c_size = int(input("Tamaño del chunk (ej: 10000): ") or "10000")
            procesar_dataset_chunks(d_in, d_out, c_size)

        elif opcion == "3":
            print("\n📖 Uso programático:")
            print("   procesar_tensor_en_gpu(tensor_cpu, modelo, batch_size=32)")

        elif opcion == "4":
            i_in = input("Carpeta con imágenes de entrada: ").strip().strip('"')
            i_out = input("Carpeta de destino: ").strip().strip('"')
            cal = int(input("Calidad JPEG (1-100, default 85): ") or "85")
            max_dim = input("Máxima dimensión en px (opcional): ").strip()
            max_dim = int(max_dim) if max_dim else None
            comprimir_imagenes_directorio(i_in, i_out, cal, max_dim)

        elif opcion == "5":
            z_in = input("Ruta del archivo original: ").strip().strip('"')
            z_out = input("Ruta del archivo comprimido: ").strip().strip('"')
            comprimir_zstd(z_in, z_out)

        elif opcion == "6":
            z_in = input("Ruta del archivo comprimido: ").strip().strip('"')
            z_out = input("Ruta del archivo destino: ").strip().strip('"')
            descomprimir_zstd(z_in, z_out)

        elif opcion == "7":
            print("\n📊 Memoria disponible:")
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / 1e9
                mem_reserv = torch.cuda.memory_reserved() / 1e9
                print(f"   GPU VRAM asignada:  {mem_alloc:.2f} GB")
                print(f"   GPU VRAM reservada: {mem_reserv:.2f} GB")
            else:
                print("   CUDA no disponible")

        elif opcion == "8":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida. Intenta nuevamente.")


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    menu()
