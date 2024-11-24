import argparse
import os
import logging
from tqdm import tqdm
import whisper

# Configuração de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constantes globais
SUPPORTED_EXTENSIONS = (".m4a", ".mp3", ".wav")
ASR_MODEL = whisper.load_model("base")


def transcribe_audio(file_path):
    """
    Transcreve um arquivo de áudio usando o modelo Whisper.

    Args:
        file_path (str): Caminho do arquivo de áudio.

    Returns:
        str: Transcrição do áudio ou mensagem de erro.
    """
    try:
        response = ASR_MODEL.transcribe(file_path)
        return response.get("text", "")
    except Exception as e:
        logging.error(f"Erro ao processar {file_path}: {e}")
        return f"Erro ao processar o arquivo: {e}"


def process_audio_files(input_dir, output_dir):
    """
    Processa todos os arquivos de áudio no diretório de entrada e salva as transcrições no diretório de saída.

    Args:
        input_dir (str): Diretório contendo arquivos de áudio.
        output_dir (str): Diretório onde os arquivos transcritos serão salvos.
    """
    # Verifica e cria o diretório de saída se necessário
    os.makedirs(output_dir, exist_ok=True)

    # Lista os arquivos suportados no diretório de entrada
    audio_files = [
        file for file in os.listdir(input_dir)
        if file.endswith(SUPPORTED_EXTENSIONS)
    ]

    if not audio_files:
        logging.warning(f"Nenhum arquivo suportado encontrado no diretório: {input_dir}")
        return

    # Itera e processa cada arquivo de áudio
    for filename in tqdm(audio_files, desc="Convertendo áudios para textos...", unit="file"):
        input_audio_path = os.path.join(input_dir, filename)
        output_text_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".txt")

        # Transcreve e salva o resultado
        transcription = transcribe_audio(input_audio_path)
        with open(output_text_path, "w", encoding="utf-8") as output_file:
            output_file.write(transcription)

    logging.info("Conversão concluída com sucesso.")


def main():
    """
    Função principal para configuração de argumentos e execução do programa.
    """
    parser = argparse.ArgumentParser(description="Transcritor de áudios para texto usando Whisper.")
    parser.add_argument("input_dir", type=str, help="Diretório contendo arquivos de áudio.")
    parser.add_argument("--output_dir", type=str, default="arquivos_transcritos",
                        help="Diretório de saída para os arquivos transcritos.")

    args = parser.parse_args()

    # Valida se o diretório de entrada existe
    if not os.path.isdir(args.input_dir):
        logging.error(f"Diretório de entrada não encontrado: {args.input_dir}")
        return

    process_audio_files(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
