# Migrating from sd-rip

The binary is now `honkbox` (`sd-rip` is gone); old paths and `SD_RIP_*`
vars are no longer read. Move your state by hand to the new default
location:

```bash
D="${XDG_STATE_HOME:-$HOME/.local/state}/honkbox"
mkdir -p "$D"
mv ~/.honkbox/jobs "$D/jobs"
mv ~/.honkbox/logs "$D/logs"
mv ~/.honkbox/models "$D/models"              # if present
mv ~/.honkbox/registry.sqlite3 "$D/"           # if present
mv ~/.honkbox/honkbox.toml "$D/"
# pre-rename installs used the same layout under ~/.sd-rip — move those
# files instead if that is where your state lives.
```

Or set `--data-dir`/`HONKBOX_DATA_DIR` to an existing `~/.honkbox` instead of
moving anything.
