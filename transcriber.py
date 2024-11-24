import argparse

import whisper
import os

from tqdm import tqdm

ASR_MODEL = whisper.load_model("base")


def make_conversion(input_audio):
    try:
        response = ASR_MODEL.transcribe(input_audio)
        return response["text"]
    except Exception as e:
        return f"Um erro inesperado ocorreu: {e}"


def convert_audio_to_text(input_dir, output_dir):
    # Verifica se o diretório de saída existe, caso contrário, cria
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(input_dir):
        print(f"Diretório de entrada {input_dir} não encontrado.")
        return

    # Itera sobre todos os arquivos no diretório de entrada
    for filename in tqdm(os.listdir(input_dir), desc=f"Convertendo audios para textos...", unit="file"):
        if filename.endswith((".m4a", ".mp3", ".wav")):
            # Caminho completo para o arquivo de entrada
            input_audio = os.path.join(input_dir, filename)

            # Nome do arquivo de txt de saida
            output_txt_file = os.path.join(output_dir, filename.replace(".m4a", ".txt"))
            output_txt = make_conversion(input_audio)
            with open(output_txt_file, "w", encoding="utf-8") as file:
                file.write(output_txt)

            # print(output_txt)
    print("Conversão concluída.")


if __name__ == "__main__":
    # Argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Trancritor de audioa")
    parser.add_argument("input_dir", type=str, help="Diretório contendo arquivos de audio")
    parser.add_argument("--output_dir", type=str, default="arquivos_transcritos",
                        help="Diretório de saída com os áudios transcritos.")

    args = parser.parse_args()
    convert_audio_to_text(args.input_dir, args.output_dir)
