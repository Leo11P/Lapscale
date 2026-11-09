Exact package versions per Mufasa environment (`main.txt`, `sam3.txt`, `da3.txt`), written by the first
`bash setup_env.sh <env>` on the cluster and then reused by every rebuild. Commit them after the first build.
`REPIN=1 bash setup_env.sh <env>` resolves the environment again and rewrites its file.
