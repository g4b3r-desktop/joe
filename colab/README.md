# Joe Colab API

Este diretório contém o backend executado dentro de uma sessão Google Colab.

## Arquivos

- Joe_Colab_API.ipynb: notebook de instalação, inicialização e publicação do túnel temporário.
- api_server.py: API FastAPI com autenticação Bearer e inferência Pocket TTS / Parakeet Redux.
- requirements.txt: dependências Python do runtime, incluindo o extra `quantize` do Pocket TTS.
- drive_model_sync.py: baixa somente os arquivos portugueses ausentes e atualiza o ZIP do Drive com validação e limite de 4 GB.
- [ADDING_TTS_MODELS.md](ADDING_TTS_MODELS.md): guia para integrar outros idiomas/modelos ao Colab, incluindo sincronização, YAML, API, testes e limite do ZIP.

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
- POST /v1/jobs/tts: multipart com text, language, model_variant opcional e campo de arquivo voice opcional; retorna job_id. Se model_variant for omitido, o servidor usa a versão padrão do idioma.
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

## Modelos no Google Drive

O notebook espera o arquivo compactado:

~~~text
Meu Drive/3d_models_ia/modelos-voz/JOE_MODELS.zip
~~~

Dentro do ZIP, o notebook mantém as duas variantes portuguesas (a estrutura pode estar diretamente na raiz ou dentro de uma pasta `JOE_MODELS/`):

~~~text
pocket-tts/
└── languages/
    ├── portuguese/model.safetensors
    └── portuguese_24l/model.safetensors
pocket-tts-without-voice-cloning/
└── languages/
    ├── portuguese/tokenizer.json
    └── portuguese_24l/tokenizer.json
~~~

Ao clicar em **Aplicar pastas e modelos**, o notebook verifica os quatro arquivos (dois checkpoints e dois tokenizers). Se todos estiverem válidos, não baixa nada. Se faltar algum ou estiver inválido, baixa somente esse arquivo da revisão oficial correspondente e atualiza o ZIP do Drive, preservando o restante do conteúdo. O ZIP é gravado apenas se o resultado validado não ultrapassar **4 GB**; em caso de erro ou excesso, o arquivo anterior é preservado.

Depois, o notebook extrai o ZIP para `/content/joe/models/` no armazenamento local da sessão Colab e gera dois YAMLs independentes: `portuguese_from_drive.yaml` e `portuguese_24l_from_drive.yaml`. A API seleciona o YAML correspondente a `model_variant`. O Pocket TTS não é carregado diretamente de dentro do ZIP. A extração precisa ser repetida quando uma nova sessão Colab for iniciada, pois `/content` é temporário.

O campo **Arquivo ZIP** permite alterar o caminho do arquivo. O notebook verifica se o ZIP existe e se contém a pasta `pocket-tts`; também aceita uma pasta externa `JOE_MODELS/`. O caminho do tokenizer é preenchido automaticamente com base na estrutura esperada, mas pode ser ajustado no campo correspondente.

Para clonagem personalizada, o Joe também pode enviar um áudio de referência em cada solicitação. O arquivo WAV ou estado `.safetensors` indicado em **Voz padrão** é opcional e precisa existir no Drive. O campo **Cache HF/ASR** define `HF_HOME`; use uma pasta persistente no Drive se quiser manter o cache entre sessões.


## Garantia de carregamento local do Pocket TTS

O notebook configura `weights_path` para o checkpoint de clonagem em `pocket-tts/languages/portuguese/model.safetensors` e o tokenizer para o arquivo local do Drive. Se existir, também configura o checkpoint de fallback sem clonagem em `pocket-tts-without-voice-cloning/languages/portuguese/model.safetensors`. A API é impedida de iniciar em modo Drive se a configuração local não tiver sido aplicada.

A voz de referência padrão não é mais um endereço `hf://`. Se o Joe não enviar um WAV na requisição, configure um WAV local no campo **Voz padrão WAV**. Os arquivos de voz em formato WAV são áudio de referência; não são os pesos do modelo.

### Sincronização automática dos modelos no ZIP do Drive

Ao aplicar a configuração do Drive, o notebook verifica os quatro arquivos das variantes portuguesas padrão e 24L:

- `pocket-tts/languages/portuguese/model.safetensors`
- `pocket-tts-without-voice-cloning/languages/portuguese/tokenizer.json`
- `pocket-tts/languages/portuguese_24l/model.safetensors`
- `pocket-tts-without-voice-cloning/languages/portuguese_24l/tokenizer.json`

As origens e revisões estão fixadas de acordo com os YAMLs oficiais do Pocket TTS. Se o ZIP não existir, ele será criado. Se algum arquivo estiver ausente, duplicado ou inválido, somente o arquivo necessário será baixado e substituído. Os demais itens existentes no arquivo são preservados.

O tamanho máximo do ZIP atualizado é **4 GB (4.000.000.000 bytes)**. O ZIP temporário é validado antes da substituição; se o tamanho final ultrapassar o limite ou ocorrer um erro, o ZIP anterior não é substituído.

Como os repositórios do modelo exigem aceitar as condições de acesso, aceite os termos em <https://huggingface.co/kyutai/pocket-tts> e crie um token Hugging Face com permissão de leitura. Salve-o no painel **Secrets** do Colab como `HF_TOKEN` e habilite **Notebook access**. O token não é gravado no Drive, no ZIP nem no notebook. Se os quatro arquivos já estiverem válidos, o download e o token não serão necessários.

A variante `portuguese_24l` usa os próprios pesos, tokenizer e YAML; não compartilha os arquivos da versão padrão. Estados de voz `.safetensors` preparados para 24L devem ser usados com a variante 24L. A rotina fica em `colab/drive_model_sync.py`, com testes em `tests/test_colab_drive_model_sync.py`.
