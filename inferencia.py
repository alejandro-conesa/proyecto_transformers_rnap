import argparse
import glob
import os
import re
import sys

import torch
from tokenizers import BertWordPieceTokenizer

from modelo import IMDBModel
from datos import limpiar


def buscar_ultimo_checkpoint(directorio: str = "weights") -> str | None:
    patron = os.path.join(directorio, "**", "*.ckpt")
    checkpoints = glob.glob(patron, recursive=True)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def cargar_modelo(checkpoint_path: str, device: torch.device) -> IMDBModel:
    print(f"  Cargando pesos desde: {checkpoint_path}")
    model = IMDBModel.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    return model


def preparar_input(
    texto: str,
    tokenizer: BertWordPieceTokenizer,
    max_len: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer.encode(texto)

    ids  = encoded.ids[:max_len]
    mask = [1] * len(ids)

    pad_len = max_len - len(ids)
    ids  += [0] * pad_len
    mask += [0] * pad_len

    return {
        "input_ids":      torch.tensor([ids],  dtype=torch.long, device=device),
        "attention_mask": torch.tensor([mask], dtype=torch.long, device=device),
    }


def clasificar(texto: str, model: IMDBModel, tokenizer, max_len: int, device: torch.device) -> dict:
    texto_limpio = limpiar(texto)
    batch = preparar_input(texto_limpio, tokenizer, max_len, device)

    with torch.no_grad():
        logits = model(batch["input_ids"], batch["attention_mask"])  # (1, 2)
        probs  = torch.softmax(logits, dim=-1).squeeze()             # (2,)

    pred  = probs.argmax().item()
    label = "POSITIVA" if pred == 1 else "NEGATIVA"

    return {
        "etiqueta":         label,
        "confianza":        probs[pred].item(),
        "prob_positiva":    probs[1].item(),
        "prob_negativa":    probs[0].item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Clasificador de sentimientos IMDB")
    parser.add_argument(
        "--checkpoint", "-c",
        default=None,
        help="Ruta al archivo .ckpt (por defecto: el más reciente en weights/)",
    )
    parser.add_argument(
        "--tokenizer", "-t",
        default="./mi_tokenizer",
        help="Carpeta con vocab.txt del tokenizer (por defecto: ./mi_tokenizer)",
    )
    parser.add_argument(
        "--max_len", "-m",
        type=int,
        default=256,
        help="Longitud máxima de tokens (debe coincidir con el entrenamiento)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Dispositivo: {device}")

    ckpt_path = args.checkpoint or buscar_ultimo_checkpoint()
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        print(
            "[ERROR] No se encontró ningún checkpoint.\n"
            "  • Entrena primero con main.py, o\n"
            "  • Indica la ruta con --checkpoint <ruta.ckpt>"
        )
        sys.exit(1)

    vocab_path = os.path.join(args.tokenizer, "vocab.txt")
    if not os.path.isfile(vocab_path):
        print(
            f"[ERROR] No se encontró el vocabulario en '{vocab_path}'.\n"
            "  Asegúrate de que la carpeta del tokenizer es correcta (--tokenizer <ruta>)."
        )
        sys.exit(1)

    print(f"  Cargando tokenizer desde: {args.tokenizer}")
    tokenizer = BertWordPieceTokenizer(vocab_path, lowercase=True)

    model = cargar_modelo(ckpt_path, device)
    max_len = args.max_len

    print("\n" + "─" * 55)
    print("Clasificador de sentimientos de reseñas IMDB")
    print("─" * 55)
    print("  Escribe una reseña en inglés y pulsa Enter.")
    print("  Escribe 'salir' o presiona Ctrl+C para terminar.\n")

    while True:
        try:
            review = input("📝 Reseña: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n¡Hasta luego!")
            break

        if not review:
            print("Escribe algo antes de pulsar Enter.\n")
            continue

        if review.lower() in {"salir", "exit", "quit", "q"}:
            print("¡Hasta luego!")
            break

        resultado = clasificar(review, model, tokenizer, max_len, device)

        print(f"\n  Sentimiento : {resultado['etiqueta']}")
        print(f"  Confianza   : {resultado['confianza'] * 100:.1f}%")
        print(f"  P(positiva) : {resultado['prob_positiva'] * 100:.1f}%  |  "
              f"P(negativa) : {resultado['prob_negativa'] * 100:.1f}%")
        print()


if __name__ == "__main__":
    main()