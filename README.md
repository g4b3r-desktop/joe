# Joe — Audio Studio

Aplicativo desktop local para **texto → áudio** e **áudio → texto**, com interface em português.

## Modelos

- **TTS — [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts):** gera WAV localmente, roda na CPU e inclui suporte a português, espanhol, francês, italiano, alemão e neerlandês, além de inglês. Permite usar uma gravação própria como referência de voz.
- **STT — [Moondream Parakeet Redux](https://huggingface.co/moondream/parakeet-redux):** transcrição local em 25 idiomas, com arquivos de pesos de aproximadamente 178 MB. Quando a implementação Photon compatível está disponível, ela é usada. Em CPUs x86 sem AVX-512 VNNI, o app tenta a implementação de referência com Transformers.

Não há envio de áudio para serviços de nuvem. Na primeira execução, os pesos são baixados do Hugging Face e ficam no cache local para usos seguintes.

## Hardware e compatibilidade

O aplicativo foi pensado para Linux desktop e também pode ser executado em Windows/macOS com Python compatível.

**Perfil de referência: Linux Mint, Core i5-2400, 16 GB RAM e RTX 3060.** O Pocket TTS roda na CPU. Para ASR, o Joe usa Photon quando o processador oferece AVX-512 VNNI; caso contrário, usa a implementação de referência Transformers e escolhe CUDA/FP16 automaticamente quando há PyTorch com CUDA disponível. Sem CUDA disponível, usa CPU.

A implementação Photon do Parakeet Redux usa kernels AVX-512 VNNI em x86. O i5-2400 oferece AVX, mas não AVX-512; por isso, para aproveitar sua RTX 3060, instale o modo CUDA. A implementação de referência do Parakeet também documenta execução em CUDA com FP16.

## Instalação no Linux Mint / Ubuntu

Requisitos: Python 3.11, 3.12 ou 3.13, python3-venv e bibliotecas de áudio do sistema.

~~~bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip libsndfile1
git clone https://github.com/g4b3r-desktop/joe.git
cd joe
~~~

Crie o ambiente e instale a build CPU do PyTorch:

~~~bash
bash scripts/install-linux.sh
~~~

Para usar a RTX 3060 na transcrição Parakeet (PyTorch CUDA 13.2), use:

~~~bash
bash scripts/install-linux.sh cuda
~~~

O Pocket TTS continua na CPU; o Parakeet usa CUDA/FP16 automaticamente quando disponível. A instalação CUDA ocupa mais espaço em disco.

Inicie:

~~~bash
bash run.sh
~~~

Também é possível instalar manualmente:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[asr-fallback]"
python -m joe
~~~

## Uso

### Texto → Áudio
1. Digite ou cole o texto.
2. Selecione o idioma da voz (português é o padrão).
3. Opcionalmente, selecione um WAV/MP3 de referência para clonagem de voz.
4. Clique em **Gerar áudio**, escolha onde salvar o WAV e aguarde o processamento.
5. Use **Reproduzir** para ouvir o resultado.

### Áudio → Texto
1. Selecione um arquivo de áudio.
2. Opcionalmente, habilite marcações de tempo por segmento.
3. Clique em **Transcrever**.
4. Revise/edite o texto e copie ou salve em TXT. O botão SRT fica disponível quando o backend fornece segmentos com tempos.

Formatos aceitos dependem do decodificador disponível no ambiente; WAV é o formato mais previsível. Parakeet Redux faz detecção automática do idioma.

## Cache e privacidade

Os modelos são baixados uma vez e armazenados nos caches padrão do Hugging Face/PyTorch. O áudio selecionado e os resultados permanecem na máquina. A seleção de um arquivo de voz para TTS só é usada localmente como condicionamento; não é enviado para uma API remota.

## Desenvolvimento e testes

~~~bash
source .venv/bin/activate
python -m pytest
ruff check .
~~~

## Estado

MVP inicial: interface desktop, geração WAV, transcrição de arquivo, cópia/salvamento de texto, exportação SRT quando o backend fornece timestamps e seleção automática de backend de ASR. Gravação direta do microfone, fila de tarefas e empacotamento AppImage/DEB ficam para etapas posteriores.

## Licenças dos modelos

- Pocket TTS: consulte a licença no repositório oficial do Kyutai.
- Parakeet Redux: CC-BY-4.0, conforme o model card. Preserve atribuição e avisos de licença ao redistribuir pesos.
