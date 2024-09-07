To enable the git hooks from this folder, type the following command in the root folder of the repository:
$ git config core.hooksPath .githooks

To disable them:
$ git config --unset core.hooksPath

To disable any hooks for one commit only:
$ git commit --no-verify