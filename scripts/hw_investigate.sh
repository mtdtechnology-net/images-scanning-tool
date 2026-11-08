#!/usr/bin/env bash
# Read-only hardware investigation for the verifier plan. Changes nothing.
# Run on the GPU server as root:  bash hw_investigate.sh 2>&1 | tee hw_report.txt
set -u
sec() { echo; echo "===== $* ====="; }

sec "CPU visibility"
nproc; nproc --all
lscpu | grep -E '^(CPU\(s\)|On-line|Off-line|Model name|Hypervisor|Virtualization type|Thread|Core|Socket)'
grep -c ^processor /proc/cpuinfo
grep -m1 -o hypervisor /proc/cpuinfo || echo "no hypervisor flag"
systemd-detect-virt || true
cat /proc/cmdline   # look for maxcpus= nr_cpus= isolcpus=

sec "cgroup limits (root + service + ollama)"
cat /sys/fs/cgroup/cpu.max 2>/dev/null; cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null
for u in scanning-tool ollama; do
  echo "--- $u"
  systemctl show $u.service -p CPUQuota -p CPUQuotaPerSecUSec -p AllowedCPUs -p CPUAffinity -p MainPID -p Environment
  cg=/sys/fs/cgroup/system.slice/$u.service
  cat $cg/cpu.max $cg/cpuset.cpus.effective 2>/dev/null
  pid=$(systemctl show $u.service -p MainPID --value)
  [ "$pid" != "0" ] && grep Cpus_allowed_list /proc/$pid/status && taskset -cp $pid
done

sec "GPU idle"
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
ollama ps
ollama list

sec "Where PaddleOCR runs + per-page timings (last 300 log lines)"
journalctl -u scanning-tool --no-pager -n 300 | grep -E 'PaddleOCR loaded|OCR page|LLM page' | tail -40

echo
echo "Now start a live scan from the web app and, in a 2nd shell, run for its duration:"
echo "  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -lms 500"
echo "  watch -n1 ollama ps        # PROCESSOR column must read 100% GPU"
echo "  top -b -d1 -n30 | grep -E 'python|ollama|Cpu'"
