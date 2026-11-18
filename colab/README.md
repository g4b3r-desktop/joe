# Joe Colab API

Este diretório contém o backend executado dentro de uma sessão Google Colab.

## Arquivos

- Joe_Colab_API.ipynb: notebook de instalação, inicialização e publicação do túnel temporário.
- api_server.py: API FastAPI com autenticação Bearer e inferência Pocket TTS / Parakeet Redux.
- requirements.txt: dependências Python do runtime.

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
- POST /v1/tts: multipart com text, language e campo de arquivo voice opcional; retorna áudio WAV.
- POST /v1/transcribe: multipart com arquivo audio e campo timestamps (none, segment ou word); retorna texto, segmentos, backend e dispositivo.

Todas as rotas são autenticadas. Swagger, ReDoc e OpenAPI foram desativados.

## Segurança operacional

O notebook cria um token aleatório novo a cada sessão. Não grave o token em células, arquivos do repositório, variáveis persistentes do Drive ou screenshots compartilhados. A URL do Quick Tunnel também é temporária. O serviço aceita uploads até 80 MB e texto até 20 mil caracteres. O túnel público é para desenvolvimento/uso pessoal; não é uma implantação de produção.
