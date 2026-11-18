# Joe Audio Studio — referência funcional

Consulte este documento ao alterar a interface ou os fluxos existentes. Verifique o código atual antes de assumir que uma opção permanece igual.

## Interface TTS

A aba Texto → Áudio mantém o fluxo de uso frequente:
- Editor do texto original.
- Botão **Gerar áudio WAV**.
- Botão **Reproduzir último áudio**.
- Botão **★ Favoritos**.
- Botão **⚙ Configurações**.
- Mensagem de estado e painel de velocidade.

As opções que não mudam a todo momento ficam no diálogo de configurações:
- **Texto e pronúncia:** normalização e dicionário de pronúncia; prévia do texto preparado.
- **Idioma e voz:** idioma da fala, caminho da voz de referência, pasta da biblioteca, abrir biblioteca e preparar voz.
- **Favoritos:** adicionar/remover a voz selecionada e editar o apelido.
- **Arquivos e atalho:** pasta de saída, nome automático e comprimento dos blocos do atalho.

O texto original não deve ser alterado pela normalização. A geração deve usar o texto preparado como entrada do modelo, preservando o texto exibido/editável.

## Biblioteca e favoritos de voz

- A biblioteca varre recursivamente a pasta raiz escolhida pelo usuário.
- Formatos conhecidos para seleção: WAV, MP3, FLAC, M4A, OGG, AAC e SAFETENSORS.
- O filtro por tipo deve limitar a lista ao formato escolhido; a busca por nome/caminho deve continuar funcionando em conjunto.
- Um clique em áudio reproduz a seleção atual. As setas Up/Down também mudam o item e iniciam a prévia.
- Arquivos `.safetensors` representam estado de voz, não áudio: não tentar reproduzi-los; parar a prévia anterior ao navegar para um estado.
- Duplo clique ou ação explícita de confirmação seleciona a voz. Ao abrir a biblioteca, restaurar e destacar o caminho selecionado anteriormente caso ainda esteja dentro da biblioteca.
- A tela principal oferece um seletor de favoritos. Os apelidos são editáveis e devem permanecer associados ao caminho canônico do arquivo.
- A persistência dos favoritos fica separada em `src/joe/voice_favorites.py`; mantenha caminhos estáveis e ignore dados corrompidos sem derrubar o app.
- Preparar uma gravação de referência com Pocket TTS exporta um arquivo `.safetensors`. Ele pode ser carregado como estado local; não é reproduzível como áudio.

## TTS local e geração por blocos

- O backend local é Pocket TTS, com geração WAV.
- A referência de voz condiciona a voz gerada. O serviço mantém cache do modelo e de estados de voz para evitar reprocessamento desnecessário.
- O atalho usa divisão de texto por sentenças/palavras e gera trechos sequencialmente.
- O áudio progressivo deve permitir ouvir trechos já gerados, enquanto a geração pode continuar. Ao terminar, o app pode usar o WAV combinado.
- A união de WAV normaliza sample rate e número de canais e pode inserir uma pausa curta entre blocos.
- Não exclua os arquivos temporários antes de o player/worker terminar de usá-los.

## Medição de velocidade

As métricas devem ser calculadas no desktop, pelo módulo `audio_performance.py`, tanto na aba principal quanto na janela compacta do atalho:
- tempo de parede gasto na operação;
- duração do WAV final;
- velocidade efetiva = duração do áudio ÷ tempo gasto, em múltiplos do tempo real;
- caracteres processados por segundo.

Durante a geração, apresentar tempo decorrido. No modo remoto, o tempo total inclui upload, fila e espera/polling da API. No modo local, inclui o carregamento inicial do modelo quando necessário. Não apresente como benchmark puro de inferência se o cronômetro inclui essas etapas.

## Atalho global e janela compacta

- Atalho documentado: `Ctrl+Shift+Espaço`.
- No Xfce, `scripts/install-desktop.sh` tenta registrar o atalho nativo; não substitua uma associação que pertence a outro aplicativo.
- `pynput` é alternativa. Wayland pode limitar captura global.
- `local_ipc.py` encaminha comandos por socket AF_UNIX com permissões restritas para a instância já aberta.
- O atalho deve iniciar a geração e mostrar apenas a janela compacta; não deve abrir uma segunda janela principal.
- A janela compacta deve ter ícone Joe e controles de progresso/reprodução/pausa/salvar.
- Ao fechar a janela, remover áudio temporário e cancelar/limpar o worker com segurança. Um arquivo salvo pelo usuário deve permanecer.
- A janela compacta pode reproduzir trechos antes de a geração completa; evite duplicar ou reiniciar áudio por conexões de eventos repetidas.

## Áudio → Texto (ASR)

- O modelo é Parakeet Redux.
- Quando suportado, o serviço usa Photon no caminho CPU; caso contrário usa Transformers.
- O caminho Transformers pode usar CPU ou CUDA com preferência explícita. Se CUDA falhar, o serviço pode tentar CPU; comunique o backend efetivo ao usuário.
- Os resultados incluem texto, backend e nota de execução. Segmentos com timestamps dependem do backend.
- O app deve permitir revisar/editar, copiar e salvar a transcrição. SRT só deve estar habilitado quando houver segmentos com tempos válidos.

## Backend remoto Google Colab

- O notebook `colab/Joe_Colab_API.ipynb` inicializa os modelos e uma API FastAPI dentro do runtime.
- O acesso usa URL HTTPS de Cloudflare Quick Tunnel e token Bearer. A URL é temporária; após reiniciar o runtime pode mudar.
- O app monitora conectividade enquanto o modo remoto está selecionado, mas esse monitor não mantém a sessão acordada.
- Endpoints documentados: `GET /health`, `POST /v1/jobs/tts`, `POST /v1/jobs/transcribe`, `GET /v1/jobs/{job_id}` e `GET /v1/jobs/{job_id}/result`.
- As rotas exigem `Authorization: Bearer <token>`. A API limita texto e uploads e utiliza uma fila de inferência; confirme os valores concretos em `colab/api_server.py` antes de alterar documentação ou cliente.
- O cliente envia arquivos de voz como upload no job TTS. `.safetensors` é estado local e não deve ser tratado como gravação original para o endpoint remoto.
- Valide respostas e formatos do áudio. O resultado TTS esperado é um WAV com cabeçalho RIFF/WAVE.
- Não invente garantia de disponibilidade, GPU ou duração da sessão do Colab.

## Normalização de fala

O módulo `text_normalization.py` mantém uma cópia preparada do texto e contém regras dependentes do idioma para números, datas, horários, moeda, porcentagens, abreviações e siglas. O dicionário pessoal tem precedência sobre regras automáticas. Ao alterar normalização:
- preserve o conteúdo original;
- mantenha o comportamento correto por idioma;
- cubra sinais, separadores decimais/milhar e datas inválidas;
- adicione testes de regressão para entradas reais;
- não “corrija” uma falha antiga sem verificar todos os idiomas afetados.


### Histórico e arquivos temporários

- Não registre no Histórico o caminho temporário da janela compacta do clipboard: ele é apagado quando a janela fecha.
- Quando o usuário salva uma cópia permanente pela janela compacta, registre o caminho salvo e associe as métricas e a voz utilizadas.
- Ao encerrar o Joe durante uma fila, peça cancelamento cooperativo e aguarde o backend retornar; não mate uma inferência nativa à força.
- A fila deve marcar como **Gerando** somente o item que está efetivamente sendo processado. Destinos duplicados entre itens pendentes devem ser recusados para evitar sobrescrita.
