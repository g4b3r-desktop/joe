# Joe Colab API

Este diretório contém o backend executado dentro de uma sessão Google Colab.

## Arquivos

- Joe_Colab_API.ipynb: notebook de instalação, inicialização e publicação do túnel temporário.
- api_server.py: API FastAPI com autenticação Bearer e inferência Pocket TTS / Parakeet Redux.
- requirements.txt: dependências Python do runtime, incluindo o extra `quantize` do Pocket TTS.

## Teste local do servidor (desenvolvimento)

É possível testar em uma máquina com Python, PyTorch e as dependências instaladas:

~~~bash
python -m pip install -r colab/requirements.txt
export JOE_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(36))')"
cd colab
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
~~~

A inferência é carregada sob demanda. Use o header:

~~~http
Authorization: Bearer <JOE_API_TOKEN>
~~~

## Rotas

- GET /health: status do servidor, GPU e estado dos modelos.
- POST /v1/jobs/tts: multipart com text, language e campo de arquivo voice opcional; retorna job_id.
- POST /v1/jobs/transcribe: multipart com arquivo audio e campo timestamps (none, segment ou word); retorna job_id.
- GET /v1/jobs/{job_id}: consulta queued, running, completed ou failed.
- GET /v1/jobs/{job_id}/result: baixa o WAV ou recebe o JSON quando completed.

Todas as rotas são autenticadas. A inferência roda em um worker sequencial e suporta até duas tarefas ativas (uma em execução e uma aguardando). O cliente usa polling para evitar manter chamadas abertas durante o tempo de inferência e contornar o limite de leitura de resposta do túnel Cloudflare. Os resultados expiram em 30 minutos se não forem retirados. Swagger, ReDoc e OpenAPI foram desativados.

## Segurança operacional

O pacote `pocket-tts[quantize]` instala o backend de quantização utilizado por `TTSModel.load_model(quantize=True)`. O notebook instala esse extra a partir de `requirements.txt` antes de iniciar a API.

O notebook cria um token aleatório novo a cada sessão. Não grave o token em células, arquivos do repositório, variáveis persistentes do Drive ou screenshots compartilhados. A URL do Quick Tunnel também é temporária. O serviço aceita uploads até 80 MB e texto até 20 mil caracteres. O túnel público é para desenvolvimento/uso pessoal; não é uma implantação de produção.

## Modelos no Google Drive

O notebook permite montar o Google Drive e configurar os caminhos dos modelos antes de iniciar a API.

Para usar o Pocket TTS em português a partir do Drive, mantenha esta estrutura:

~~~text
MyDrive/JOE_MODELS/
├── pocket-tts/languages/portuguese/model.safetensors
├── pocket-tts-without-voice-cloning/languages/portuguese/tokenizer.json
└── huggingface-cache/
~~~

Baixe os arquivos no computador com Hugging Face CLI (após autenticar e aceitar os termos aplicáveis):

~~~bash
hf download kyutai/pocket-tts --include "languages/portuguese/**" --local-dir JOE_MODELS/pocket-tts
hf download kyutai/pocket-tts-without-voice-cloning languages/portuguese/tokenizer.json --local-dir JOE_MODELS/pocket-tts-without-voice-cloning
hf download kyutai/pocket-tts g-Vi8PgmSY0-enhanced-v2.wav --local-dir JOE_MODELS/pocket-tts
~~~

No widget do notebook, informe a pasta do repositório Pocket TTS, o caminho completo do tokenizer e, opcionalmente, um WAV ou arquivo de estado `.safetensors` para servir como voz padrão. O notebook gera um YAML local com `weights_path` e `tokenizer_path` apontando para os arquivos do Drive. Para clonagem personalizada, o Joe também pode enviar um áudio de referência em cada solicitação.

O campo **Cache HF/ASR** define `HF_HOME`; o Photon/Parakeet mantém ali o cache dos arquivos baixados na primeira execução. Assim, o cache pode persistir no Drive entre sessões. O cache do Hugging Face não é o mesmo que uma pasta de snapshot local do repositório.

A pasta do Drive precisa conter os arquivos antes de clicar em **Aplicar pastas e modelos**. Se a caixa de uso do Pocket TTS local estiver desmarcada, a API usa a configuração oficial pelo Hugging Face.
