# Joe Colab API

Este diretório contém o backend executado dentro de uma sessão Google Colab.

## Arquivos

- Joe_Colab_API.ipynb: notebook de instalação, inicialização e publicação do túnel temporário.
- api_server.py: API FastAPI com autenticação Bearer e inferência Pocket TTS / Parakeet Redux.
- requirements.txt: dependências Python do runtime, incluindo o extra `quantize` do Pocket TTS.
- drive_assets.py: valida, organiza e protege modelos/vozes persistidos em arquivos individuais no Google Drive.
- drive_model_sync.py: rotina legada do fluxo ZIP; não é importada nem executada pelo notebook de armazenamento direto.
- [ADDING_TTS_MODELS.md](ADDING_TTS_MODELS.md): guia histórico do fluxo ZIP; não é usado pelo notebook de armazenamento direto.

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
- POST /v1/jobs/tts: multipart com text, language, model_variant e voice_id opcional; no modo Drive direto, a voz deve estar preparada no Drive.
- POST /v1/assets/uploads: inicia upload em blocos de um modelo (checkpoint + tokenizer) ou áudio de referência.
- PUT /v1/assets/uploads/{upload_id}/{asset_key}/{chunk_index}: recebe um bloco de até 8 MiB.
- POST /v1/assets/uploads/{upload_id}/commit: valida e instala o modelo ou prepara/persiste a voz no Google Drive.
- GET /v1/assets: lista modelos e vozes persistidos no Drive. Se model_variant for omitido, o servidor usa a versão padrão do idioma.
- POST /v1/jobs/transcribe: multipart com arquivo audio e campo timestamps (none, segment ou word); retorna job_id.
- GET /v1/jobs/{job_id}: consulta queued, running, completed ou failed.
- GET /v1/jobs/{job_id}/result: baixa o WAV ou recebe o JSON quando completed.
- DELETE /v1/jobs/{job_id}: cancela uma tarefa aguardando execução ou solicita o descarte do resultado quando a inferência já começou.

Todas as rotas são autenticadas. A inferência roda em um worker sequencial e suporta até duas tarefas ativas (uma em execução e uma aguardando). O cliente usa polling para evitar manter chamadas abertas durante o tempo de inferência e contornar o limite de leitura de resposta do túnel Cloudflare. O cancelamento interrompe imediatamente tarefas ainda na fila; uma inferência em execução não pode ser interrompida com segurança no meio do kernel/modelo, então o servidor marca a tarefa para cancelamento e descarta o resultado assim que a chamada termina. Os resultados expiram em 30 minutos se não forem retirados. Swagger, ReDoc e OpenAPI foram desativados.

## Segurança e configuração persistente

O pacote `pocket-tts[quantize]` instala o backend de quantização utilizado por `TTSModel.load_model(quantize=True)`. O notebook instala esse extra a partir de `requirements.txt` antes de iniciar a API.

Para não redigitar URL e token a cada sessão, configure uma vez estes Secrets no painel de chave do Google Colab e habilite **Notebook access**:

- `JOE_API_URL`: URL HTTPS fixa do hostname publicado no Cloudflare Named Tunnel (por exemplo, `https://api.seudominio.com`).
- `JOE_API_TOKEN`: token Bearer forte e persistente, usado pelo desktop Joe.
- `CLOUDFLARE_TUNNEL_TOKEN`: token do Named Tunnel, diferente do token Bearer.

O notebook carrega os Secrets usando `google.colab.userdata.get()` e os disponibiliza como variáveis de ambiente somente durante o runtime. Para gerar um token Bearer forte uma única vez, execute `python -c 'import secrets; print(secrets.token_urlsafe(36))'` e salve o resultado como valor de `JOE_API_TOKEN` no painel Secrets. O Joe desktop também salva a URL e o token Bearer na configuração do usuário após a primeira inserção. Assim, com domínio fixo, a configuração é reaproveitada nas próximas sessões sem precisar copiar valores novamente.

**Não use uma URL Quick Tunnel em `JOE_API_URL` esperando que ela permaneça igual.** O Quick Tunnel cria um subdomínio aleatório e temporário; nesse modo, atualize a URL no Joe depois de reiniciar o runtime. Para manter a URL, publique um hostname fixo no Named Tunnel e direcione-o para `http://localhost:8000`.

Não grave os segredos diretamente em células, arquivos do repositório, variáveis persistentes do Drive ou screenshots compartilhados. O serviço aceita uploads até 80 MB e texto até 20 mil caracteres. O túnel público é para desenvolvimento/uso pessoal; não é uma implantação de produção.

### Variantes do Pocket TTS

O campo opcional `model_variant` aceita somente variantes compatíveis com o idioma informado. O Joe desktop envia esse campo automaticamente para que a mesma seleção funcione no modo Local e no Colab. Português, espanhol, francês, italiano, alemão e neerlandês oferecem o modelo padrão e a variante de 24 camadas (preview). Inglês também oferece versões datadas, sampler Drifting e variantes compactas de 18,2 M, 6,9 M e 1,9 M de parâmetros. O japonês está disponível somente como **experimental**, usando o modelo compacto de 6 camadas `japanese` de [takuma104/pocket-tts-ja-cc](https://huggingface.co/takuma104/pocket-tts-ja-cc) (aproximadamente 219 MB); o servidor carrega o YAML publicado pelo autor e usa a voz japonesa `female_03.wav` quando não há gravação própria. A versão japonesa de 24 camadas não é exposta. As variantes portuguesas padrão e `portuguese_24l` usam YAMLs próprios com checkpoints e tokenizers locais do Drive; os demais modelos usam seus identificadores ou configurações específicas.

## Armazenamento direto no Google Drive (sem ZIP)

O notebook atual usa a pasta persistente:

~~~text
Meu Drive/JOE_TTS/
├── models/<variante>/
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── config.yaml
│   └── manifest.json
├── voices/<variante>/<voice-id>/
│   ├── reference.<wav|flac|mp3|ogg>
│   ├── voice_state.safetensors
│   └── metadata.json
├── .uploads/
├── .staging/
└── cache/
~~~

**O fluxo do notebook não usa mais `JOE_MODELS.zip`, não extrai arquivos e não define os YAMLs portugueses a partir do ZIP.** A rotina antiga `drive_model_sync.py` permanece no repositório por compatibilidade/histórico, mas não é chamada pelo notebook novo.

### Enviar o modelo do computador

Na aba **Google Colab / API** do Joe, clique em **Enviar modelo local ao Google Drive** e escolha uma pasta contendo no nível principal:

- `model.safetensors`: pesos com suporte a clonagem.
- `tokenizer.json`: tokenizer da mesma versão do modelo.

O Joe envia os dois arquivos separadamente em blocos de 8 MiB, evitando uma requisição multipart única maior que os limites comuns de túneis HTTP. O Colab confere tamanho, SHA-256, cabeçalho Safetensors e JSON do tokenizer. Depois cria o YAML local derivado do template da mesma variante instalado no runtime e substitui a pasta do modelo atomicamente. Se a transferência ou validação falhar, a versão anterior no Drive permanece.

O upload pela interface cobre as variantes cujo YAML está disponível no pacote Pocket TTS instalado. A instalação de modelos comunitários com YAML próprio exige adaptar o resolvedor e o validador antes de ser habilitada.

### Vozes preparadas no próprio Colab

Quando existe um áudio de referência selecionado no Joe e o modo Google Colab está ativo, o cliente:

1. Envia o áudio original em blocos.
2. O Colab verifica se já existe uma voz com o mesmo hash de áudio e fingerprint do modelo atual.
3. Se não existir, carrega o modelo do Drive, chama `get_state_for_audio_prompt()` e exporta o estado com `export_model_state()`.
4. Salva a gravação original, o estado e o manifesto da voz no Drive.
5. Usa o identificador remoto da voz para todos os trechos daquela geração.

Se o modelo for substituído, o fingerprint muda. Estados preparados com pesos/tokenizer antigos são rejeitados e o áudio original precisa ser enviado novamente para preparação. O fluxo direto aceita WAV, FLAC, MP3 e OGG. Não envie um `.safetensors` criado localmente: o Colab precisa preparar o estado com os pesos que realmente estão instalados no Drive.

### Limites e segurança

- Cada bloco de upload tem 8 MiB; o cliente calcula SHA-256 antes da transferência.
- O checkpoint pode ter até 4 GB e o tokenizer até 100 MiB. O áudio de referência tem limite de 80 MiB.
- Uploads incompletos ficam em `.uploads/`; arquivos finais são instalados somente após conferência do tamanho e hash.
- Modelos e vozes são gravados em diretórios temporários e promovidos por substituição atômica.
- Não compartilhe a pasta `JOE_TTS` se ela contiver gravações privadas. Os tokens de acesso continuam sendo lidos dos Secrets do Colab, nunca dos arquivos de modelo/voz.
- O Drive permanece persistente entre sessões; o conteúdo de `/content` continua temporário.

### Testes

~~~bash
python -m pytest tests/test_colab_drive_assets.py tests/test_colab_drive_asset_client.py -q
python -m pytest -q
~~~

A rotina de upload e preparação depende de uma sessão real do Colab com Google Drive montado. Os testes automatizados verificam as validações, protocolo de blocos e proteção de compatibilidade, mas não substituem um teste de geração real no runtime.
