#!/usr/bin/env bash
# Ensure bash even if invoked via `sh`
if [ -z "${BASH_VERSION:-}" ]; then exec /usr/bin/env bash "$0" "$@"; fi
set -euo pipefail

########################################
# 사용자가 바꿀 값
BN='[Train]_[195]_[0.1800].ckpt'  # 받을 체크포인트 파일명
RUN_DATE='20250830_151751'                                      # 예: 20250816_Base 경로의 날짜 부분
########################################

# 고정/환경 값(필요시만 수정)
REMOTE_HOST='work@nipa.nhncloud.com'
REMOTE_PORT='10532'
REMOTE_ROOT='/home/work/CatVTON/logs'
LOCAL_ROOT='/home/kjk/CatVTON/logs'

# 경로 구성
REMOTE_DIR="${REMOTE_ROOT}/${RUN_DATE}_ddp/models"
LOCAL_DIR="${LOCAL_ROOT}/${RUN_DATE}_ddp/models"

REMOTE="${REMOTE_DIR}/${BN}"
DST="${LOCAL_DIR}/${BN}"

mkdir -p "$LOCAL_DIR"

ssh_base=(ssh -p "${REMOTE_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -T)

# 1) 사이즈 확인
REMOTE_SIZE="$("${ssh_base[@]}" "${REMOTE_HOST}" "env -i /usr/bin/stat -c%s '$REMOTE'" 2>/dev/null || true)"
LOCAL_SIZE=0
[ -f "$DST" ] && LOCAL_SIZE="$(stat -c%s "$DST")"

echo "REMOTE_SIZE: ${REMOTE_SIZE:-<unknown>}"
echo "LOCAL_SIZE : $LOCAL_SIZE"

if [ -z "${REMOTE_SIZE}" ]; then
  echo "WARN: 원격 파일 크기 조회 실패. 다음 실행에서 재시도하세요. 파일은 유지됩니다: $DST"
  exit 0
fi

# 2) 상태 분기
if [ "$LOCAL_SIZE" -gt "$REMOTE_SIZE" ]; then
  echo "WARN: 로컬 파일이 더 큽니다. 과거 잘못 이어받았을 수 있습니다. 파일은 유지합니다: $DST"
  echo "      필요 시 수동 정리 후 처음부터 다시 받거나 검증하세요."
  exit 0
fi

if [ "$LOCAL_SIZE" -lt "$REMOTE_SIZE" ]; then
  # 3) 이어받기: dd 우선, 실패 시 tail 폴백
  BYTES_LEFT=$(( REMOTE_SIZE - LOCAL_SIZE ))
  echo "==> resuming ${BYTES_LEFT} bytes from offset ${LOCAL_SIZE} -> $DST"

  # dd 시도 (skip_bytes로 바이트 단위 오프셋)
  set +e
  "${ssh_base[@]}" "${REMOTE_HOST}" \
    "env -i LC_ALL=C PATH=/usr/bin:/bin dd if='$REMOTE' iflag=skip_bytes skip=${LOCAL_SIZE} bs=8M status=progress" \
    >> "$DST"
  DD_RC=$?
  set -e

  if [ $DD_RC -ne 0 ]; then
    echo "DD WARN: dd 전송 실패(rc=$DD_RC). tail -c 폴백을 시도합니다."
    set +e
    # tail은 +N 번째 바이트부터 전송(N=LOCAL_SIZE+1)
    START=$(( LOCAL_SIZE + 1 ))
    "${ssh_base[@]}" "${REMOTE_HOST}" \
      "env -i LC_ALL=C PATH=/usr/bin:/bin tail -c +${START} '$REMOTE'" \
      >> "$DST"
    TAIL_RC=$?
    set -e
    if [ $TAIL_RC -ne 0 ]; then
      echo "ERROR: tail 폴백도 실패(rc=$TAIL_RC). 네트워크 상태/원격 권한을 확인하세요. 파일은 유지됩니다: $DST"
      exit 1
    fi
  fi
fi

# 4) 전송 후 사이즈 재확인
NEW_LOCAL_SIZE=0
[ -f "$DST" ] && NEW_LOCAL_SIZE="$(stat -c%s "$DST")"
echo "POST-TRANSFER LOCAL_SIZE: $NEW_LOCAL_SIZE"

if [ "$NEW_LOCAL_SIZE" -lt "$REMOTE_SIZE" ]; then
  PCT=$(( REMOTE_SIZE>0 ? (100 * NEW_LOCAL_SIZE / REMOTE_SIZE) : 0 ))
  echo "PARTIAL: ${PCT}% 다운로드됨. 다음 실행에서 자동 이어받습니다(파일 유지)."
  exit 0
fi

if [ "$NEW_LOCAL_SIZE" -gt "$REMOTE_SIZE" ]; then
  echo "WARN: 전송 후 로컬 크기가 원격보다 큽니다. 이어받기 중 오염 가능성. 파일은 유지합니다: $DST"
  exit 0
fi

# 5) 크기 동일 → SHA-256 검증
echo "==> sizes match, verifying SHA-256"
REMOTE_SHA="$("${ssh_base[@]}" "${REMOTE_HOST}" "env -i sha256sum '$REMOTE' | awk '{print \$1}'" 2>/dev/null || true)"
LOCAL_SHA="$(sha256sum "$DST" | awk '{print $1}')"

echo "REMOTE: ${REMOTE_SHA:-<unknown>}"
echo "LOCAL : $LOCAL_SHA"

if [ -z "${REMOTE_SHA}" ]; then
  echo "WARN: 원격 해시 조회 실패. 다음 실행에서 재검증하세요. 파일은 유지됩니다: $DST"
  exit 0
fi

if [ "$REMOTE_SHA" = "$LOCAL_SHA" ]; then
  echo "OK: 무결성 검증 통과. 완료 파일: $DST"
else
  echo "NG: 해시 불일치. 파일은 유지됩니다: $DST"
  echo "    필요 시 처음부터 다시 받거나 서버 측 파일을 재검증하세요."
fi
