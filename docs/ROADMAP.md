# Roadmap de melhorias — Joe Audio Studio

Todas as etapas serão implementadas nesta branch de trabalho, preservando a branch estável e a main.

## Fase 1 — Base, fila e estabilidade

- [x] Criar branch isolada feature/joe-studio-roadmap.
- [x] Separar o gerenciamento da fila TTS em módulo testável, sem dependência de Qt.
- [ ] Fila FIFO integrada à tela Texto → Áudio.
- [ ] Visualizar estados da fila, remover itens ainda não iniciados e cancelar a tarefa atual de forma cooperativa.
- [ ] Continuar as tarefas seguintes após conclusão ou falha da tarefa anterior.
- [ ] Separar gradualmente as telas e controladores que ainda vivem em app.py.
- [ ] Revisar concorrência entre geração manual, atalho do clipboard e transcrição.

**Limite conhecido do cancelamento:** não é seguro interromper à força uma chamada síncrona do Pocket TTS ou uma requisição HTTP que já está em andamento. A solicitação de cancelamento é cooperativa: impede o início se ainda não começou; durante a inferência, aguarda a chamada atual retornar, descarta o arquivo dessa tarefa e não começa outro bloco. A implementação não afirma cancelar o processamento remoto dentro do Colab.

A fila será mantida em memória nesta fase; fechar o Joe exige confirmação e descarta as tarefas que ainda aguardam. Persistência e retomada serão avaliadas em etapa posterior.

## Fase 2 — Histórico e player

- Histórico pesquisável de WAVs, com data, duração, tamanho, voz, backend e métricas.
- Associar texto original e metadados ao arquivo, com opção para não guardar o texto.
- Player com waveform, seek, volume e velocidade de reprodução.
- Operações de arquivo com confirmação e proteção contra exclusões acidentais.

## Fase 3 — Perfis de voz e transcrição avançada

- Perfis que agrupam voz, apelido, idioma e opções de pronúncia.
- Editor de transcrição sincronizado com o áudio.
- Exportação VTT e melhorias em legendas.
- Avaliar diarização e timestamps por palavra de acordo com o suporte real dos modelos.

## Fase 4 — Distribuição e endurecimento

- Gerenciamento seguro de credenciais do Colab.
- Melhorias de reconexão, diagnóstico e cancelamento remoto quando o servidor suportar.
- Pacotes Linux AppImage/DEB, verificações de atualização e documentação de migração.
- Cobertura adicional de testes e verificação manual no Linux Mint/Xfce.

## Regras de execução

- Uma fase/entrega pequena por commit; não fazer reescrita total.
- Manter compatibilidade com os rascunhos e preferências salvos em QSettings.
- Não remover modelos, caches, vozes, WAVs ou scripts antigos.
- Rodar testes focados, suíte completa, Ruff e compilação; documentar qualquer falha já existente.
- Não integrar na branch estável automaticamente.
