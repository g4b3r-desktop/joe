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

O notebook espera o arquivo compactado:

~~~text
Meu Drive/3d_models_ia/modelos-voz/JOE_MODELS.zip
~~~

Dentro do ZIP, mantenha esta estrutura (ela pode estar diretamente na raiz ou dentro de uma pasta `JOE_MODELS/`):

~~~text
pocket-tts/languages/portuguese/model.safetensors
pocket-tts-without-voice-cloning/languages/portuguese/tokenizer.json
huggingface-cache/
~~~

Ao clicar em **Aplicar pastas e modelos**, o notebook extrai o ZIP para `/content/joe/models/` no armazenamento local da sessão Colab e aponta a configuração para os arquivos extraídos. O Pocket TTS não é carregado diretamente de dentro do ZIP. A extração precisa ser repetida quando uma nova sessão Colab for iniciada, pois `/content` é temporário.

O campo **Arquivo ZIP** permite alterar o caminho do arquivo. O notebook verifica se o ZIP existe e se contém a pasta `pocket-tts`; também aceita uma pasta externa `JOE_MODELS/`. O caminho do tokenizer é preenchido automaticamente com base na estrutura esperada, mas pode ser ajustado no campo correspondente.

Para clonagem personalizada, o Joe também pode enviar um áudio de referência em cada solicitação. O arquivo WAV ou estado `.safetensors` indicado em **Voz padrão** é opcional e precisa existir no Drive. O campo **Cache HF/ASR** define `HF_HOME`; use uma pasta persistente no Drive se quiser manter o cache entre sessões.


## Garantia de carregamento local do Pocket TTS

O notebook configura `weights_path` para o checkpoint de clonagem em `pocket-tts/languages/portuguese/model.safetensors` e o tokenizer para o arquivo local do Drive. Se existir, também configura o checkpoint de fallback sem clonagem em `pocket-tts-without-voice-cloning/languages/portuguese/model.safetensors`. A API é impedida de iniciar em modo Drive se a configuração local não tiver sido aplicada.

A voz de referência padrão não é mais um endereço `hf://`. Se o Joe não enviar um WAV na requisição, configure um WAV local no campo **Voz padrão WAV**. Os arquivos de voz em formato WAV são áudio de referência; não são os pesos do modelo.
