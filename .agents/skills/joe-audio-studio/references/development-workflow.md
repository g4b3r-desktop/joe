# Joe Audio Studio — fluxo de desenvolvimento

## Atualizar o checkout local

A partir da pasta do repositório:

```bash
git status --short --branch
git fetch origin
git checkout feature/joe-studio-improvements
git pull --ff-only origin feature/joe-studio-improvements
```

Não use `git reset --hard`, `git clean -fdx`, force-push ou comandos que descartem alterações do usuário. Se houver arquivos modificados, inspecione-os e preserve-os.

## Ambiente de desenvolvimento

Instalação típica no Linux Mint/Ubuntu:

```bash
bash scripts/install-linux.sh
```

Ativar o ambiente e instalar dependências de desenvolvimento:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Execução:

```bash
bash run.sh
```

Verificações recomendadas:

```bash
python -m pytest
ruff check .
python -m compileall -q src colab
```

Para uma mudança específica, executar primeiro o teste relevante (por exemplo `python -m pytest tests/test_voice_favorites.py -q`) e depois o conjunto completo.

## Cuidados com modelos e hardware

- Não baixe modelos durante testes unitários comuns; use mocks para serviços externos e carregamento de pesos.
- Não exija CUDA para testes que devem passar em CPU/CI.
- O perfil de referência documentado inclui RTX 3060 12 GB, mas o Pocket TTS usa CPU. O ASR CUDA depende da build correta do PyTorch e do driver local.
- Não inclua `.venv`, caches Hugging Face/PyTorch, checkpoints, modelos, WAVs, arquivos de voz ou diretórios temporários em commits.
- Evite usar a GPU/VRAM de maneira concorrente sem necessidade durante teste e desenvolvimento.

## Padrões de implementação

- Siga estilo existente: type hints, funções pequenas, mensagens em português e tratamento explícito de erros.
- Configure a interface Qt no thread principal; operações lentas de modelo/rede/arquivo devem usar o mecanismo de workers existente.
- Não bloquear a GUI com inferência, downloads, polling de rede ou varreduras grandes de arquivos.
- Persistir preferências com as chaves QSettings existentes. Ao renomear uma chave, criar migração e teste em vez de perder a configuração antiga.
- Manter a lógica pura em módulos pequenos quando possível; testar sem iniciar QApplication quando não for necessário.
- Não duplicar cálculo de métricas, tratamento de favoritos, normalização de WAV ou regras de API já centralizados nos módulos.
- Validar caminhos e formatos antes de abrir arquivos; mostrar erros úteis em vez de traceback cru ao usuário.
- Nunca registrar token Colab, conteúdo integral do clipboard, transcrição privada ou texto enviado à API em logs.

## Commit e atualização do GitHub

Antes de escrever no GitHub:
1. Obter o SHA atual de `feature/joe-studio-improvements`.
2. Conferir o conteúdo atual dos arquivos que serão alterados.
3. Criar as alterações com base na árvore/commit atual.
4. Criar commit cujo pai é o head confirmado.
5. Atualizar a branch com comparação do SHA esperado, `force=false`.
6. Se o head mudou ou a atualização foi rejeitada, buscar de novo e reconciliar. Nunca sobrescrever commits que chegaram depois.

Mensagem de commit concisa, por exemplo `Add voice library extension filter`.

Na resposta, listar commits reais e verificáveis, testes realmente executados e falhas que persistem. O comando para o usuário sincronizar deve ser explícito:

```bash
git checkout feature/joe-studio-improvements
git pull --ff-only origin feature/joe-studio-improvements
.venv/bin/pip install -e .
bash run.sh
```

## Documentação

Atualize `README.md` quando mudar instalação, UX visível, formato suportado, limites, endpoints, privacidade ou configuração. Atualize o notebook e seus testes quando o contrato da API mudar. Não documente uma capacidade até que exista no código e esteja testada.
