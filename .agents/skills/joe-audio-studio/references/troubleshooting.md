# Joe Audio Studio — diagnóstico rápido

Sempre reproduza o problema no ambiente/caminho correto e examine logs sem expor texto privado, arquivos de voz ou tokens.

## Atalho global não funciona

- Confira se a sessão é X11 ou Wayland (`echo "$XDG_SESSION_TYPE"`).
- No Xfce, verifique a associação `Ctrl+Shift+Espaço` com `xfconf-query`.
- Instale/reinstale apenas o atalho gerenciado pelo Joe: `bash scripts/install-desktop.sh`.
- Não sobrescreva um comando existente de outro aplicativo.
- Em Wayland, `pynput` pode não capturar teclas globalmente; preferir o atalho nativo do desktop.
- Consulte estado do listener, servidor IPC e socket em `src/joe/local_ipc.py`.

## Erro de áudio RIFF/WAVE

WAV esperado precisa ter assinatura `RIFF` e identificador `WAVE` no cabeçalho. Não confunda gravação de referência com estado `.safetensors`. O serviço remoto precisa receber um áudio de referência que o backend consiga decodificar; `.safetensors` é formato de estado local e não áudio WAV. Verifique também se o resultado remoto realmente contém WAV antes de salvar/reproduzir.

## Voice library não reproduz um arquivo

- Confirmar se o arquivo é um formato de áudio e não `.safetensors`.
- Verificar suporte do decodificador Qt no sistema para o container/codec.
- Conferir se o caminho continua acessível e se o player não está em estado de erro.
- Não tentar reproduzir um `.safetensors`; mostrar explicação e parar a prévia anterior.

## Pocket TTS lento ou falha ao carregar

- O carregamento inicial do modelo e a codificação da voz podem afetar a primeira geração.
- Verificar cache de modelos, permissões e dependências; não excluir cache ou arquivos preparados sem autorização.
- Quando o usuário já possui um estado `.safetensors`, confirmar se o backend local o carrega diretamente.
- Para comparar velocidade, usar a medição do desktop e separar tempo inicial de carregamento do tempo de inferência quando a instrumentação permitir.

## Colab sem conexão ou job falhou

- Conferir se o runtime continua ativo, se o servidor foi iniciado e se a URL do Quick Tunnel ainda é a atual.
- Conferir token Bearer sem exibi-lo na resposta ou logs.
- O Quick Tunnel e a sessão Colab são temporários; reconectar pode exigir URL/token novos.
- HTTP 401/403 normalmente indica token recusado; HTTP 429 pode indicar fila/limite; job failed exige inspecionar mensagem do servidor.
- O monitor de conectividade não impede suspensão/desconexão do Colab.
- Não sugerir que keep-alive/pings garantem sessão ativa.

## ASR CPU/CUDA

- Conferir a preferência escolhida e o backend retornado pelo serviço.
- Photon CPU em x86 depende de AVX-512 VNNI; CPUs sem isso usam Transformers.
- CUDA depende da build de PyTorch compatível e disponibilidade de CUDA. A implementação prevê fallback para CPU em algumas falhas.
- Não diagnosticar falta de GPU apenas pelo nome do dispositivo; confirmar `torch.cuda.is_available()` e logs do serviço.
