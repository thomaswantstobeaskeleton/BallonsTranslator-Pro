# Region merge tool

The Region merge tool (Tools → Region merge tool) now includes:

- **Quick profile**: Conservative / Balanced / Aggressive presets for overlap/gap thresholds.
- **Preview current page**: runs a dry-run merge and reports block count delta without applying changes.
- **Include-only labels (whitelist)**: restrict merges to specific label classes.
- Existing blacklist, label grouping, output shape type, and merge mode controls.

## Suggested workflow

1. Start with **Conservative** profile and click **Preview current page**.
2. If under-merging, move to **Balanced** and preview again.
3. Use **Aggressive** only for dense word-fragmented pages.
4. Enable **Whitelist labels** to avoid merging non-dialog classes.

## Notes

- Preview does not mutate project data.
- Run on current page applies in-memory immediately.
- Run on all pages writes merged results to the project JSON.


## How merge settings work together

- **Detector collision merge** (Config → Text detector) runs during detection and is mainly for dense word-level fragments.
- **Region merge tool** runs after detection/pipeline (manual or auto-after-run) and can further merge existing boxes with label and geometry rules.
- If both are enabled, the effective order is: detector merge first, region merge second.
- Use **Import detector collision-merge defaults** in Region merge to start from similar aggressiveness, then tune label constraints (whitelist/blacklist/groups).
