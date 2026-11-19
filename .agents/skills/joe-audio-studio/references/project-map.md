# Joe Audio Studio — mapa do projeto

Snapshot de referência: branch `feature/joe-studio-improvements`, commit `e6ecda33e7c3104328c374899c32c33cbfcf4ee1`. Confirme os caminhos e a estrutura no checkout atual antes de usar.

## Estrutura

```text
.
├── .github/workflows/python-checks.yml
├── .agents/skills/joe-audio-studio/       # esta skill
├── colab/
│   ├── Joe_Colab_API.ipynb
│   ├── api_server.py
│   ├── README.md
│   └── requirements.txt
├── scripts/
│   ├── install-linux.sh
│   └── install-desktop.sh
├── src/joe/
│   ├── app.py
│   ├── __main__.py
│   ├── audio_chunks.py
│   ├── audio_history.py              # índice SQLite dos WAVs gerados
│   ├── audio_performance.py
│   ├── ui/
│   │   ├── __init__.py
│   │   └── audio_history.py          # painel e ações do histórico
│   ├── local_ipc.py
│   ├── text_normalization.py
│   ├── utils.py
│   ├── voice_favorites.py
│   ├── resources/joe.png
│   └── services/
│       ├── pocket_tts.py
│       ├── parakeet.py
│       └── colab_api.py
├── tests/
│   ├── test_audio_chunks.py
│   ├── test_audio_history.py
│   ├── test_audio_performance.py
│   ├── test_colab_api.py
│   ├── test_colab_notebook.py
│   ├── test_local_ipc.py
│   ├── test_text_normalization.py
│   ├── test_utils.py
│   └── test_voice_favorites.py
├── pyproject.toml
└── README.md
```

## Responsabilidades

| Arquivo | Responsabilidade |
|---|---|
| `src/joe/app.py` | Janela principal PySide6, abas, diálogos, formulários, preferências QSettings, players, workers, atalho e fluxo da aplicação |
| `src/joe/__main__.py` | Entrada do módulo; encaminha ativações/atalho à instância existente antes de importar a GUI pesada |
| `src/joe/services/pocket_tts.py` | Adaptador local do Pocket TTS, cache do modelo/estado de voz, criação de WAV e exportação de estado `.safetensors` |
| `src/joe/services/parakeet.py` | Adaptador Parakeet Redux; Photon quando compatível, Transformers como alternativa, CPU/CUDA |
| `src/joe/services/colab_api.py` | Cliente HTTP remoto autenticado, submissão de jobs e polling dos resultados |
| `colab/api_server.py` | API FastAPI, autenticação Bearer, fila de inferência, limites e resultados dos jobs |
| `colab/Joe_Colab_API.ipynb` | Preparação do runtime, acesso opcional ao Drive, monitoramento e túnel temporário |
| `src/joe/audio_chunks.py` | Divisão por sentenças e união de trechos WAV com taxa de amostragem/canais normalizados |
| `src/joe/audio_history.py` | Persistência SQLite dos metadados dos áudios; texto original é opcional e vem desativado |
| `src/joe/ui/audio_history.py` | Aba de histórico, filtros, reprodução, renomear, exportar, excluir e gerenciar textos opcionais |
| `src/joe/audio_performance.py` | Métricas compartilhadas nas telas: duração, tempo total, razão tempo real e caracteres por segundo |
| `src/joe/local_ipc.py` | Servidor/cliente AF_UNIX para comandos locais da instância ativa |
| `src/joe/voice_favorites.py` | Canonicalização de caminhos, leitura e serialização dos apelidos favoritos |
| `src/joe/text_normalization.py` | Conversão de números, datas, horas, moeda, porcentagem, abreviações e siglas para fala |
| `scripts/install-linux.sh` | Ambiente virtual, PyTorch CPU/CUDA e instalação editável |
| `scripts/install-desktop.sh` | `joe.desktop`, ícone, atalho Xfce e desinstalação segura dos itens gerenciados pelo Joe |

## Stack e requisitos

- Python `>=3.11,<3.14` (projeto usa Python 3.11–3.13).
- PySide6 para a interface desktop.
- Pocket TTS para texto → fala.
- Parakeet Redux para áudio → texto.
- NumPy, SciPy e SoundFile para tratamento de áudio.
- HTTPX para o cliente remoto; FastAPI/Uvicorn no servidor Colab.
- pytest e Ruff como ferramentas de desenvolvimento.

O README descreve como perfil de referência Linux Mint/Xfce, Core i5-2400, 16 GB RAM e RTX 3060 12 GB. Isso é um perfil documentado, não um requisito universal. Pocket TTS roda na CPU. Para ASR, a opção CUDA usa PyTorch CUDA quando disponível; Photon em CPU requer suporte de instruções específico (AVX-512 VNNI em x86).

## Configuração e dados do usuário

- A aplicação usa `QSettings` para restaurar preferências e campos entre sessões.
- Não sobrescreva ou limpe configurações existentes sem uma migração explícita.
- Pastas de modelos, gravações, saídas WAV e caches pertencem ao usuário; não as inclua no Git.
- O token Colab é armazenado nas configurações locais sem criptografia na implementação documentada. Nunca exponha o valor em logs, relatórios, screenshots ou commits.
