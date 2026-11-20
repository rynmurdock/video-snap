
This is an experimental generative art codebase with tooling initially focused on running generative models over segmented video.



Some important notes about how we contribute to this repo:

Code is written to be clear, modular, & self-explanatory. In-line comments are kept to two lines at most. Docs are action-oriented: they are for tl;drs & copy/pasting lines.

Smoke tests are vital for fully new work or work that could break functionality. Tests that save time by checking essential functionality quickly are good (e.g., "can we import everything & run a skeleton model?") but we do not right tests that "only might" come in handy.

We tend to use functions instead of objects for simplicity: we're building pipes, not houses here.

Functions within this codebase take in & return video & images as cv2 numpy arrays. They do not output or take in any other formats -- when they use other formats, it is as required & internal to the functions for some sort of transferring between libaries. This adds some simplicity & ease to architecting & extending the codebase.

Our models are setup to run on an rtx 4090; this is subject to change but is the modus operandi for now.

After you've made a significant change (new feature, fix, etc. but *not* some e.g. three-line change or edit to docs/comments), add it to your worktree, push the changes, then run tests & start a backgrounded subagent audit. Whenever you push, give me the branchname so I can diff it on my side.

Try to keep your responses brief & to-the-point.

When you provide info like branch names, bold them. 
