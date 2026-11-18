# Joe — Audio Studio

Aplicativo desktop para **texto → áudio** e **áudio → texto**, com interface em português e dois modos de execução: **Local** ou **Google Colab (API remota)**.

## Atalho global de texto para áudio

No Linux Mint/Xfce, o instalador registra `Ctrl+Shift+Espaço` como atalho nativo do Xfce. Ele funciona com a janela do Joe minimizada ou em segundo plano: o comando abre um processo auxiliar, envia a solicitação à instância aberta por um socket local privado e encerra o auxiliar. O áudio é gerado na janela compacta do Joe.

O Joe mantém o listener global `pynput` como alternativa quando o atalho nativo do Xfce não está instalado. O listener é monitorado; se encerrar, o status informa a falha e **Arquivo → Reativar atalho global** permite tentar novamente. Em Wayland, a captura global pelo `pynput` é limitada; o atalho nativo do desktop é preferível.

O instalador não substitui um comando existente em `Ctrl+Shift+Espaço`. Para registrar novamente o atalho após atualizar, execute `bash scripts/install-desktop.sh`. Para remover somente o atalho criado pelo Joe, use `bash scripts/install-desktop.sh --uninstall`; comandos que não pertencem ao Joe são preservados.

## Modelos

- **TTS — [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts):** gera WAV localmente, roda na CPU e inclui suporte a português, espanhol, francês, italiano, alemão e neerlandês, além de inglês. Instala o extra `quantize` recomendado pelo projeto e permite usar uma gravação própria como referência de voz.
- **STT — [Moondream Parakeet Redux](https://huggingface.co/moondream/parakeet-redux):** transcrição local em 25 idiomas, com pesos compactados de aproximadamente 178 MB. Quando a implementação Photon compatível está disponível, ela é usada. Em CPUs sem AVX-512 VNNI, o app usa a implementação de referência Transformers; em FP32 ela pode expandir o uso de memória para cerca de 2,5 GB.

No modo Local, o processamento ocorre no próprio computador. No modo Colab, texto e arquivos de áudio selecionados são enviados pela conexão HTTPS ao runtime temporário do Google Colab para inferência. Os pesos são baixados no ambiente em que o modelo estiver sendo executado e permanecem no cache daquele runtime enquanto ele existir.


## Modo Google Colab (API remota)

O Joe pode usar uma sessão do Google Colab como servidor de inferência, sem instalar os pesos dos modelos no computador desktop.

1. Abra a aba **Google Colab / API** no Joe e clique em **Abrir notebook no Google Colab** (ou abra [o notebook diretamente](https://colab.research.google.com/github/g4b3r-desktop/joe/blob/feature/colab-model-api/colab/Joe_Colab_API.ipynb)).
2. No Colab, selecione um runtime com GPU quando disponível e execute as células em sequência.
3. A última célula inicia a API FastAPI e um Cloudflare Quick Tunnel temporário. Ela mostra uma URL HTTPS `trycloudflare.com` e um token Bearer aleatório.
4. Cole a URL e o token na aba **Google Colab / API**, teste a conexão e selecione **Google Colab (API remota)** em Modo de execução.
5. Use as abas **Texto → Áudio** e **Áudio → Texto** normalmente. Os resultados continuam sendo salvos/revisados no desktop.

Enquanto **Google Colab (API remota)** estiver selecionado e a URL/token estiverem preenchidos, o Joe verifica a conectividade a cada 60 segundos e mostra o estado da API. Essa verificação é apenas monitoramento: não mantém o runtime acordado nem evita os limites de inatividade do Colab. Se a sessão terminar, execute novamente o notebook e informe a nova URL do Quick Tunnel e o token. O Joe voltará a verificar a conexão automaticamente.

Para continuar trabalhando sem depender da sessão remota, selecione **Neste computador (Local)** no modo de execução. O Joe já possui os serviços locais de Pocket TTS e Parakeet Redux; o processamento local não depende do Colab, embora exija os modelos instalados/cacheados no computador. A troca para o modo local é manual para evitar enviar silenciosamente tarefas para outro backend.

O notebook usa Pocket TTS na CPU e Parakeet Redux via Photon CUDA quando a GPU do runtime está disponível. Sem GPU, usa a implementação de referência Transformers em CPU, que pode ser consideravelmente mais lenta.

### Endpoints

A API autenticada oferece:

| Método | Endpoint | Função |
|---|---|---|
| GET | `/health` | Estado da sessão, CUDA e modelos carregados |
| POST | `/v1/jobs/tts` | Multipart com `text`, `language` e `voice` opcional; retorna um ID de tarefa |
| POST | `/v1/jobs/transcribe` | Multipart com `audio` e `timestamps`; retorna um ID de tarefa |
| GET | `/v1/jobs/{job_id}` | Consulta o estado da inferência |
| GET | `/v1/jobs/{job_id}/result` | Retorna WAV ou JSON quando a tarefa termina |

Todas as rotas exigem `Authorization: Bearer <token>`. A inferência usa uma fila assíncrona (um worker de modelo e até duas tarefas ativas), então o cliente consulta o estado em vez de manter uma requisição aberta durante a geração/transcrição. Resultados não resgatados expiram após 30 minutos. A API não habilita páginas públicas Swagger/OpenAPI e limita uploads de áudio a 80 MB e texto a 20.000 caracteres.

### Limitações e segurança

- O Quick Tunnel é temporário e a URL muda ao reiniciar a sessão. O Colab pode desconectar o runtime por inatividade, limites de uso ou duração máxima; a disponibilidade de GPU não é garantida.
- A URL e o token devem ser tratados como credenciais. Não publique a saída do notebook nem compartilhe o token. O Joe mantém o token apenas na memória durante a execução e grava somente a URL/modo nas configurações.
- O modo Colab envia o texto e/ou o áudio para o runtime do Colab através do túnel HTTPS. Não use esse modo para conteúdo que não possa sair do computador.
- Feche a sessão quando terminar. O notebook inclui uma célula para encerrar o servidor e o túnel.
- O Quick Tunnel é voltado a desenvolvimento/uso pessoal temporário; não é um endpoint de produção com disponibilidade garantida.

Consulte [as limitações oficiais do Colab](https://research.google.com/colaboratory/faq.html) e a [documentação de Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/).

## Hardware e compatibilidade

O aplicativo foi pensado para Linux desktop e também pode ser executado em Windows/macOS com Python compatível.

**Perfil de referência: Linux Mint, Core i5-2400, 16 GB RAM e RTX 3060.** O Pocket TTS roda na CPU. Para ASR, o Joe oferece seleção explícita entre CPU e NVIDIA GPU. No modo CPU, usa Photon se o processador for compatível; caso contrário, usa Transformers. No modo CUDA, usa a implementação de referência Transformers em FP16 quando a build CUDA do PyTorch está instalada.

A implementação Photon do Parakeet Redux usa kernels AVX-512 VNNI em x86. O i5-2400 oferece AVX, mas não AVX-512; por isso, selecione NVIDIA GPU (CUDA) para usar a RTX 3060. O modelo também documenta execução em CUDA com FP16. O modo CPU é útil quando outros modelos já estão ocupando VRAM.

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

Ao concluir, o instalador registra automaticamente **Joe Audio Studio** no menu de aplicativos do seu usuário, criando `~/.local/share/applications/joe.desktop`. Você poderá abrir o programa pelo menu do Linux Mint, sem precisar abrir um terminal.

Para também criar um ícone `joe.desktop` na área de trabalho:

~~~bash
bash scripts/install-linux.sh --desktop
~~~

Para usar a RTX 3060 na transcrição Parakeet (PyTorch CUDA 13.2), use:

~~~bash
bash scripts/install-linux.sh cuda
~~~

O atalho pode ser instalado ou recriado sem reinstalar as dependências:

~~~bash
bash scripts/install-desktop.sh
~~~

Para criar também o ícone na área de trabalho, use `bash scripts/install-desktop.sh --desktop`. Para remover somente os atalhos do menu e da área de trabalho, sem apagar o programa ou os modelos:

~~~bash
bash scripts/install-desktop.sh --uninstall
~~~

O lançador chama o `run.sh` da pasta onde o repositório foi instalado, usando o ambiente virtual `.venv`. Se você mover a pasta do Joe, execute novamente `bash scripts/install-desktop.sh` para atualizar o caminho do `joe.desktop`.

O Pocket TTS continua na CPU. Na aba Áudio → Texto, selecione **NVIDIA GPU (CUDA)** para usar a RTX; escolha **CPU (preserva a VRAM)** quando estiver executando outros modelos. A instalação CUDA ocupa mais espaço em disco.

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

# Para usar CUDA 13.2 no Parakeet, substitua a instalação do torch por:
python -m pip install torch --index-url https://download.pytorch.org/whl/cu132
~~~

## Uso

### Áudio rápido do texto copiado

1. Copie qualquer texto no sistema, em um navegador, editor ou outro aplicativo.
2. Pressione **Ctrl+Shift+Espaço**. O Joe captura o clipboard e começa a gerar o áudio automaticamente, sem precisar colar o texto na aba Texto → Áudio.
3. Uma pequena janela do Joe aparece enquanto o áudio é gerado. Quando terminar, use **Reproduzir**, **Pausar** ou **Salvar áudio…**.
4. Ao fechar essa janela, o WAV temporário é apagado. Quando você usa **Salvar áudio…**, uma cópia permanente fica no local escolhido.

Esse atalho usa o idioma, a voz, a normalização, o dicionário de pronúncia e o modo Local/Google Colab já selecionados no Joe. O texto do clipboard só é lido quando o atalho é acionado. Também existe a ação **Arquivo → Gerar áudio do texto copiado** como alternativa.

O atalho global usa pynput. Em sessões Linux X11 (como a sessão Xfce padrão do Linux Mint) ele pode capturar teclas fora da janela do Joe. Em Wayland, o ambiente gráfico pode bloquear a captura global; nesse caso, use a ação pelo menu Arquivo.

### Salvamento automático de rascunhos e campos
O Joe grava automaticamente, enquanto você digita ou altera os campos, os textos do Texto → Áudio, a transcrição e sua edição, os caminhos de arquivos, a pasta de saída, idioma, dispositivo, opções, URL e token do Colab e a última aba aberta. Esses dados são restaurados ao iniciar o aplicativo, mesmo que você não tenha gerado áudio ou iniciado uma transcrição. O token do Colab é armazenado nas configurações locais do usuário sem criptografia; proteja o arquivo de configuração e sua conta do sistema.

### Texto → Áudio
1. Digite ou cole o texto. A caixa de texto mantém o conteúdo original.
2. Por padrão, o Joe normaliza números, datas, horários, porcentagens, valores em reais, abreviações e siglas antes de enviar uma cópia ao Pocket TTS. Essa normalização automática pode ser desativada sem desligar o dicionário personalizado.
3. Use **Prévia do texto falado** para conferir a cópia preparada sem alterar o texto original.
4. Use **Dicionário de pronúncia** para cadastrar, editar ou remover palavras e expressões personalizadas. As regras são salvas nas configurações do usuário e são aplicadas antes das regras automáticas.
5. Selecione o idioma da voz (português é o padrão).
6. Defina **Pasta da biblioteca de vozes** para apontar à pasta principal onde estão suas gravações e estados. O Joe procura recursivamente em todas as subpastas; em **Biblioteca…**, filtre pelo nome/caminho, selecione a voz e use **Reproduzir prévia** antes de confirmar. Também é possível ouvir a voz já selecionada em **Ouvir voz**. WAV/MP3/FLAC/M4A/OGG/AAC podem ser reproduzidos; `.safetensors` é um estado processado e não contém áudio reproduzível.
7. Para reutilizar a mesma voz sem reprocessar o áudio de referência em cada nova sessão, selecione o WAV/MP3 e clique em **Preparar voz (.safetensors)…**. Estados `.safetensors` são aceitos no modo local; no modo Colab remoto, selecione o áudio original em WAV (RIFF/WAVE), formato exigido pelo endpoint atual. Escolha onde salvar o estado. O Joe passa a usar o arquivo `.safetensors` selecionado como voz padrão. O processamento do áudio original acontece apenas na preparação; nas sessões seguintes, o Pocket TTS carrega o estado salvo diretamente.
8. A pasta padrão já é **Música/Joe**. Você pode alterá-la uma vez no campo Pasta para salvar; o Joe memoriza a escolha para as próximas execuções.
9. A opção **Salvar automaticamente com nome genérico** fica ativada por padrão. Os arquivos recebem nomes como `joe_audio_20261118_123045.wav`, sem abrir uma janela para escolher nome ou local a cada geração. Nomes repetidos no mesmo segundo recebem um sufixo para não sobrescrever arquivos.
10. Desative a opção automática quando quiser escolher manualmente o nome/local de um áudio específico e use **Reproduzir** para ouvir o resultado.
11. A medição de velocidade é feita pelo próprio Joe no computador, sem depender de métricas calculadas no notebook. Na aba **Texto → Áudio** e na janela compacta do atalho, o aplicativo mostra o tempo decorrido durante a geração; ao concluir, apresenta tempo total, duração do WAV, velocidade efetiva em múltiplos do tempo real (áudio produzido ÷ tempo gasto) e caracteres processados por segundo. No modo remoto, o cronômetro do desktop inclui envio, fila e espera pela resposta da API; no modo local, inclui o carregamento do modelo quando necessário. Os números representam o tempo percebido de ponta a ponta.

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
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
~~~

## Estado

MVP inicial: interface desktop, geração WAV, transcrição de arquivo, cópia/salvamento de texto, exportação SRT quando o backend fornece timestamps, seleção local CPU/CUDA e integração com API do Google Colab. Gravação direta do microfone, fila de tarefas e empacotamento AppImage/DEB ficam para etapas posteriores.

## Licenças dos modelos

- Código do Pocket TTS: MIT no repositório oficial; confira também a licença CC-BY-4.0 indicada para os pesos do modelo.
- Parakeet Redux: CC-BY-4.0, conforme o model card. Preserve atribuição e avisos de licença ao redistribuir pesos.
