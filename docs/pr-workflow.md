# Branch and pull request workflow

```bash
git checkout main && git pull
git checkout -b feature/<short-name>
# make and test changes
git add <files>
git commit -m "Clear, specific message"
git push -u origin feature/<short-name>
```

Open a pull request, describe what changed, link the issue with `Closes #<issue>`, request a teammate review, address comments, and merge after approval. Each team member should make meaningful commits and contribute to issues and reviewed pull requests. The personal-repository branch `capstone/final-submission` is being prepared for review; merge only after checking its file contents.
