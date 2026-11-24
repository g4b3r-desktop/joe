# Joe — Audio Studio

Aplicativo desktop para **texto → áudio** e **áudio → texto**, com interface em português e dois modos de execução: **Local** ou **Google Colab (API remota)**.

## Atalho global de texto para áudio

No Linux Mint/Xfce, o instalador registra `Ctrl+Shift+Espaço` como atalho nativo do Xfce. Ele funciona com a janela do Joe minimizada ou em segundo plano: o comando abre um processo auxiliar, envia a solicitação à instância aberta por um socket local privado e encerra o auxiliar. O áudio é gerado na janela compacta do Joe.

O Joe mantém o listener global `pynput` como alternativa quando o atalho nativo do Xfce não está instalado. O listener é monitorado; se encerrar, o status informa a falha e **Arquivo → Reativar atalho global** permite tentar novamente. Em Wayland, a captura global pelo `pynput` é limitada; o atalho nativo do desktop é preferível.

O instalador não substitui um comando existente em `Ctrl+Shift+Espaço`. Para registrar novamente o atalho após atualizar, execute `bash scripts/install-desktop.sh`. Para remover somente o atalho criado pelo Joe, use `bash scripts/install-desktop.sh --uninstall`; comandos que não pertencem ao Joe são preservados.

## Modelos

- **TTS — [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts):** gera WAV localmente, roda na CPU e inclui suporte nativo a português, inglês, espanhol, francês, italiano, alemão e neerlandês. O japonês está disponível como opção **experimental**, por meio do modelo comunitário compacto de 6 camadas [takuma104/pocket-tts-ja-cc](https://huggingface.co/takuma104/pocket-tts-ja-cc), com aproximadamente 219 MB de pesos. O Joe baixa configuração, pesos e voz de referência japonesa no primeiro uso, armazenando o cache em `.models/huggingface/`. A versão japonesa de 24 camadas não aparece no seletor. O modelo exige Pocket TTS 3.3.0 ou superior e usa licença CC BY 4.0; atribuição: takuma104, Pocket TTS Japanese (CC BY 4.0), baseado em Kyutai Pocket TTS.
- **STT — [Moondream Parakeet Redux](https://huggingface.co/moondream/parakeet-redux):** transcrição local em 25 idiomas, com pesos compactados de aproximadamente 178 MB. Quando a implementação Photon compatível está disponível, ela é usada. Em CPUs sem AVX-512 VNNI, o app usa a implementação de referência Transformers; em FP32 ela pode expandir o uso de memória para cerca de 2,5 GB.

No modo Local, o processamento ocorre no próprio computador. No modo Colab, texto e arquivos de voz selecionados (gravações WAV/MP3 ou estados `.safetensors`) são enviados pela conexão HTTPS ao runtime temporário do Google Colab para inferência. Os pesos são baixados no ambiente em que o modelo estiver sendo executado e permanecem no cache daquele runtime enquanto ele existir.


## Modo Google Colab (API remota)

O Joe pode usar uma sessão do Google Colab como servidor de inferência, sem instalar os pesos dos modelos no computador desktop.

1. Abra a aba **Google Colab / API** no Joe e clique em **Abrir notebook no Google Colab** (ou abra [o notebook diretamente](https://colab.research.google.com/github/g4b3r-desktop/joe/blob/feature/joe-studio-improvements/colab/Joe_Colab_API.ipynb)).
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
- O modo Colab envia o texto e os arquivos de voz selecionados (incluindo estados `.safetensors`) para o runtime do Colab através do túnel HTTPS. Não use esse modo para conteúdo que não possa sair do computador.
- Feche a sessão quando terminar. O notebook inclui uma célula para encerrar o servidor e o túnel.
- O Quick Tunnel é voltado a desenvolvimento/uso pessoal temporário; não é um endpoint de produção com disponibilidade garantida.

Consulte [as limitações oficiais do Colab](https://research.google.com/colaboratory/faq.html) e a [documentação de Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/).

## Hardware e compatibilidade

O aplicativo foi pensado para Linux desktop e também pode ser executado em Windows/macOS com Python compatível.

**Perfil de referência: Linux Mint, Core i5-2400, 16 GB RAM e RTX 3060.** O Pocket TTS permite escolher CPU ou NVIDIA GPU (CUDA) nas configurações de Texto → Áudio. Na CPU, o usuário pode ativar ou desativar a quantização dinâmica INT8; ela vem ativada por padrão para preservar memória, mas pode ser desligada para carregar pesos sem quantização. Na CUDA, a quantização INT8 fica indisponível e é automaticamente ignorada, pois o Pocket TTS só oferece esse caminho de quantização na CPU. A preferência da CPU é preservada ao alternar entre dispositivos. Também é possível ajustar a quantidade de threads usadas pelo PyTorch na CPU. A comparação **Comparar CPU × CUDA** executa o mesmo texto, modelo e voz nos dois dispositivos e mostra separadamente o carregamento do modelo, preparação da voz, tempo médio de geração, velocidade em tempo real e pico de VRAM. Também existem perfis **Econômico** (CPU + INT8 + uma thread), **Equilibrado** (CPU + INT8 + aproximadamente metade das threads) e **Desempenho** (prioriza CUDA; sem CUDA usa CPU sem INT8 e todas as threads). Qualquer opção pode ser ajustada manualmente depois de aplicar um perfil. O painel de monitoramento atualiza CPU, RAM, utilização da GPU, VRAM e temperatura NVIDIA a cada 2,5 segundos, quando a ferramenta `nvidia-smi` estiver disponível. O histórico registra dispositivo, threads, quantização e perfil usados, sem salvar texto adicional por causa desse recurso. O resultado depende do modelo e do hardware; CUDA não é necessariamente mais rápida em todos os computadores. Para ASR, o Joe oferece seleção independente entre CPU e NVIDIA GPU. No modo CPU, usa Photon se o processador for compatível; caso contrário, usa Transformers. No modo CUDA, usa a implementação de referência Transformers em FP16 quando a build CUDA do PyTorch está instalada.

A implementação Photon do Parakeet Redux usa kernels AVX-512 VNNI em x86. O i5-2400 oferece AVX, mas não AVX-512; por isso, selecione NVIDIA GPU (CUDA) para usar a RTX 3060. O modelo também documenta execução em CUDA com FP16. O modo CPU é útil quando outros modelos já estão ocupando VRAM.

## Instalação no Linux Mint / Ubuntu

Requisitos: Python 3.11, 3.12 ou 3.13, python3-venv e bibliotecas de áudio do sistema.

~~~bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip libsndfile1 shared-mime-info xdg-utils
git clone https://github.com/g4b3r-desktop/joe.git
cd joe
~~~

Crie o ambiente e instale a build CPU do PyTorch:

~~~bash
bash scripts/install-linux.sh
~~~

Ao concluir, o instalador registra automaticamente **Joe Audio Studio** no menu de aplicativos do seu usuário, criando `~/.local/share/applications/joe.desktop`. Você poderá abrir o programa pelo menu do Linux Mint, sem precisar abrir um terminal. Ele também registra a extensão `.joeaudio`: depois da instalação, dê duplo clique em um projeto salvo para iniciar o Joe e abrir o conteúdo. Se o Joe já estiver em execução, o arquivo é encaminhado para a instância aberta.

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

Para o **Parakeet (Áudio → Texto)**, selecione **NVIDIA GPU (CUDA)** para usar a RTX; escolha **CPU (preserva a VRAM)** quando estiver executando outros modelos. Para o **Pocket TTS (Texto → Áudio)**, abra **Configurações → Hardware e desempenho do Pocket TTS** para alternar entre CPU e CUDA e ajustar as threads da CPU. A instalação CUDA ocupa mais espaço em disco.

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

Em **Configurações → Arquivos, histórico e atalho de texto copiado**, ative **Frases clicáveis** para gerar cada frase separadamente. Na janela compacta, clique em uma frase pronta para ouvir somente aquele trecho; o botão principal continua reproduzindo o WAV completo. Frases muito longas são divididas respeitando o limite de caracteres configurado. Como cada frase exige uma chamada TTS própria, esse modo pode demorar mais em textos com muitas frases.

Opcionalmente, ative **Reconhecer a fala após gerar**. Depois do TTS, o Parakeet Redux transcreve o WAV completo. Quando o backend retornar timestamps, os segmentos reconhecidos aparecem em uma segunda lista e podem ser reproduzidos isoladamente. O backend local Transformers pode retornar somente o texto, sem tempos; nesse caso, a lista de frases TTS continua disponível. No modo Colab, os timestamps dependem do backend Parakeet disponível no runtime. O reconhecimento é desligado por padrão para não acrescentar tempo de processamento a cada atalho.

O atalho global usa pynput. Em sessões Linux X11 (como a sessão Xfce padrão do Linux Mint) ele pode capturar teclas fora da janela do Joe. Em Wayland, o ambiente gráfico pode bloquear a captura global; nesse caso, use a ação pelo menu Arquivo.

### Salvamento automático de rascunhos e campos
O Joe grava automaticamente, enquanto você digita ou altera os campos, os textos do Texto → Áudio, a transcrição e sua edição, os caminhos de arquivos, a pasta de saída, idioma, dispositivo, opções, URL e token do Colab e a última aba aberta. Esses dados são restaurados ao iniciar o aplicativo, mesmo que você não tenha gerado áudio ou iniciado uma transcrição. O token do Colab é armazenado nas configurações locais do usuário sem criptografia; proteja o arquivo de configuração e sua conta do sistema.

### Texto → Áudio
1. Digite ou cole o texto. A caixa de texto mantém o conteúdo original.
2. Use **⚙ Configurações** na aba Texto → Áudio para acessar as opções menos usadas. Elas ficam separadas da tela principal e são salvas automaticamente.
3. Em **Texto e pronúncia**, ative ou desative a normalização de números, datas, horários, porcentagens, valores em reais, abreviações e siglas. Use **Prévia do texto falado** para conferir a cópia preparada sem alterar o texto original.
4. Em **Texto e pronúncia → Dicionário de pronúncia**, cadastre, edite ou remova palavras e expressões personalizadas. Use **Importar JSON…** para selecionar um ou vários arquivos com centenas ou milhares de termos; as entradas são mescladas ao dicionário atual e termos repetidos (sem diferenciar maiúsculas/minúsculas) têm a pronúncia atualizada. Você pode filtrar a lista por termo ou pronúncia. A importação aceita um objeto simples como `{"cpu":"cê pê u","gpu":"gê pê u"}`, um objeto agrupado em `entries` ou uma lista de itens `term`/`pronunciation`; clique em **Salvar** para gravar as alterações. Em **Regras avançadas…**, crie regras independentes com idioma específico ou todos os idiomas, correspondência por palavra inteira, expressão com espaços flexíveis ou trecho dentro de palavra, diferenciação de maiúsculas/minúsculas, ativação individual e prioridade (-1000 a 1000). Regras de maior prioridade são avaliadas primeiro e a pronúncia substituta não é alterada pelas regras automáticas. A **Prévia do texto falado** mostra o resultado final sem modificar o texto original.
5. Em **Idioma e voz**, selecione o idioma da fala (português é o padrão) e configure sua voz de referência. O japonês aparece como **experimental**: usa o modelo compacto de 219 MB e, sem voz personalizada, utiliza uma voz japonesa sintética incluída no repositório do modelo. Para esse idioma, a normalização automática de números, datas e valores é desativada para preservar kanji e kana; o dicionário e as regras de pronúncia personalizadas continuam disponíveis. Em **Perfis de voz**, salve combinações nomeadas de arquivo de voz, idioma e normalização; selecione um perfil e clique em **Aplicar perfil** para restaurar os três ajustes. Salvar ou excluir perfis não altera favoritos, arquivos de voz nem o dicionário de pronúncia. No grupo **Favoritos**, digite um apelido e clique em **Adicionar aos favoritos**; o botão **★ Favoritos** na tela principal permite localizar, ouvir e selecionar rapidamente as vozes salvas. O apelido pode ser alterado e salvo novamente, ou removido dos favoritos.
6. Defina a **Pasta da biblioteca** para apontar à pasta principal onde estão suas gravações e estados. O Joe procura recursivamente em todas as subpastas. Em **Biblioteca…**, filtre pelo nome/caminho e escolha o formato no seletor (Todos, WAV, MP3, FLAC, M4A, OGG, AAC ou SAFETENSORS). Um clique em um arquivo de áudio ou a navegação entre itens com as setas **↑/↓** inicia a reprodução automaticamente; use **Parar reprodução** quando necessário. Dê duplo clique ou pressione **Usar esta voz** para selecionar o arquivo. WAV/MP3/FLAC/M4A/OGG/AAC podem ser reproduzidos; `.safetensors` é um estado processado e não contém áudio reproduzível.
7. Para reutilizar a mesma voz sem reprocessar o áudio de referência em cada sessão, selecione o WAV/MP3 e clique em **Preparar voz (.safetensors)…**. O Pocket TTS aceita os estados `.safetensors` tanto no modo local quanto no modo Google Colab; o Joe envia o arquivo preservando a extensão para que o runtime carregue o estado preparado. No modo remoto, o conteúdo do arquivo de voz é enviado ao Colab pelo túnel HTTPS.
8. Em **Arquivos e atalho de texto copiado**, altere a pasta de saída, escolha o salvamento automático com nome genérico e ajuste o tamanho dos blocos usados pelo atalho global.
9. A medição de velocidade é feita pelo próprio Joe no computador, sem depender de métricas calculadas no notebook. Na tela Texto → Áudio e na janela compacta do atalho, o aplicativo mostra o tempo decorrido durante a geração; ao concluir, apresenta tempo total, duração do WAV, velocidade efetiva em múltiplos do tempo real (áudio produzido ÷ tempo gasto) e caracteres processados por segundo. No modo remoto, o cronômetro do desktop inclui envio, fila e espera pela resposta da API; no modo local, inclui o carregamento do modelo quando necessário.

### Fila de geração de áudio

O botão **Fila de geração** abre um painel para adicionar o texto atual, acompanhar o progresso por tarefa, cancelar itens, marcar tarefas urgentes, repetir falhas e limpar tarefas concluídas. As gerações TTS são executadas em sequência; uma falha é registrada na fila e não impede o processamento das próximas tarefas. Cada tarefa captura o texto preparado, idioma, voz, destino e backend escolhidos no momento do envio.

A fila existe somente na memória da sessão atual. Os textos não são gravados pela fila nem exibidos na lista; os arquivos WAV concluídos ficam no destino escolhido. O cancelamento local é cooperativo entre blocos de até 280 caracteres: uma inferência que já começou precisa terminar antes de o worker parar. No Colab, tarefas ainda aguardando podem ser canceladas; se a inferência remota já começou, o servidor sinaliza o cancelamento e descarta o resultado quando a chamada terminar.

### Histórico de áudios

A aba **Histórico** registra os WAV gerados pelo TTS da tela principal e os áudios do atalho de clipboard somente quando você os salva em uma pasta permanente. Ela mostra nome, data, duração, tamanho, voz/apelido, versão do modelo Pocket TTS, tempo de geração, velocidade efetiva e origem (Local ou Google Colab). Use pesquisa e filtros por voz, data e duração para localizar arquivos; os comandos permitem reproduzir, abrir a pasta, renomear, exportar uma cópia ou excluir o WAV e seu registro.

O índice é um banco SQLite local na pasta de dados do Joe. Os WAVs permanecem nas pastas escolhidas pelo usuário; não são movidos para uma biblioteca interna. Arquivos antigos já existentes não são indexados automaticamente. A opção **Configurações → Arquivos, histórico e atalho → Privacidade do histórico** controla se o texto original também será guardado: vem desativada por padrão. Com a opção desligada, ficam salvos apenas os metadados do áudio; com ela ligada, o texto pode ser pesquisado no histórico. Desativar a opção impede guardar texto em novos registros, mas não apaga textos que já tenham sido armazenados. Na aba Histórico, **Ver texto salvo** permite consultar/copiar o texto de um item; **Apagar textos salvos** remove todos os textos antigos do banco, preservando os WAVs e seus metadados. A opção não altera os rascunhos já existentes nas configurações do Joe.

### Áudio → Texto
1. Selecione um arquivo de áudio.
2. Opcionalmente, habilite marcações de tempo por segmento.
3. Clique em **Transcrever**.
4. Revise/edite o texto e copie ou salve em TXT. O botão SRT fica disponível quando o backend fornece segmentos com tempos.

Formatos aceitos dependem do decodificador disponível no ambiente; WAV é o formato mais previsível. Parakeet Redux faz detecção automática do idioma.


Quando o backend retorna segmentos com tempos, o Joe mostra a lista de trechos sincronizados abaixo dos controles. Clique em um trecho para posicionar o player no seu início ou use **Reproduzir com sincronização** para iniciar o áudio junto com a transcrição. Durante a reprodução, o trecho correspondente é realçado automaticamente. A sincronização depende de timestamps fornecidos pelo backend; se eles não estiverem disponíveis, a transcrição normal e a edição do texto continuam funcionando.

### Player avançado

A aba **Player avançado** controla o mesmo reprodutor usado na geração, na prévia da biblioteca de vozes e no histórico. Inclui waveform com navegação por clique, barra de progresso, velocidade de 0,75× a 2×, volume, saltos de 5 e 10 segundos e repetição de trechos delimitados pelos marcadores A e B.

Para cortar o início ou o fim, marque A e/ou B e use **Exportar corte**. Sem marcadores, os limites padrão são o início e o fim do arquivo. A detecção e a redução de silêncio permitem ajustar o nível em dB, a duração mínima da pausa e quanto silêncio manter. As operações exportam um novo WAV e não sobrescrevem o arquivo original.

A leitura da waveform e o processamento são executados em segundo plano. A waveform depende dos formatos decodificáveis pela biblioteca de áudio instalada; a reprodução continua usando o decodificador Qt disponível no sistema.

## Cache e privacidade

No modo **Local**, os modelos e arquivos de voz do Pocket TTS são baixados automaticamente na primeira utilização e armazenados dentro da pasta do projeto: `.models/huggingface/` (com os pesos do Hub em `.models/huggingface/hub/`). Assim, os downloads acompanham a instalação do Joe e não se misturam ao cache global do Hugging Face do usuário. A pasta `.models/` é ignorada pelo Git e não será enviada ao GitHub. Para liberar espaço, feche o Joe e remova essa pasta; os modelos serão baixados novamente quando forem usados. No modo **Google Colab**, os downloads continuam no cache temporário do runtime do Colab.

O áudio selecionado e os resultados permanecem na máquina. A seleção de um arquivo de voz para TTS só é usada localmente como condicionamento; não é enviado para uma API remota.

## Desenvolvimento e testes

~~~bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
~~~

## Estado

MVP: interface desktop, geração WAV em fila sequencial com progresso por tarefa, cancelamento cooperativo, prioridade urgente e repetição de falhas, transcrição de arquivo, cópia/salvamento de texto, exportação SRT quando o backend fornece timestamps, seleção local CPU/CUDA e integração com API do Google Colab. Gravação direta do microfone e empacotamento AppImage/DEB ficam para etapas posteriores.

## Licenças dos modelos

- Código do Pocket TTS: MIT no repositório oficial; confira também a licença CC-BY-4.0 indicada para os pesos do modelo.
- Parakeet Redux: CC-BY-4.0, conforme o model card. Preserve atribuição e avisos de licença ao redistribuir pesos.
