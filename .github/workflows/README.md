# GitHub Actions Workflows

This directory contains CI/CD workflows for the Arxiv Curator project.

## Workflows

### CI (`ci.yml`)

Runs on every push and pull request to main branches.

**Jobs:**
- **test**: Runs linting, formatting checks, tests, and builds the wheel
- **validate-notebooks**: Validates Databricks notebook syntax

**Triggers:**
- Push to `main`, `lecture5`, or `feat/*` branches
- Pull requests to `main` or `lecture5`

### CD (`cd.yml`)

Deploys the application to Databricks.

**Jobs:**
- **build-and-deploy**: Builds wheel, uploads to DBFS, deploys notebooks and resources, triggers model registration job

**Triggers:**
- Push to `main` branch
- Tags matching `v*`
- Manual workflow dispatch (allows choosing environment)

## Setup

### Required Secrets

Add these secrets to your GitHub repository settings:

1. **DATABRICKS_HOST**: Your Databricks workspace URL (e.g., `https://adb-123456789.azuredatabricks.net`)
2. **DATABRICKS_TOKEN**: Personal access token or service principal token with permissions to:
   - Upload files to DBFS
   - Import notebooks to workspace
   - Run jobs

### Creating Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each secret with its value

### Environment Configuration (Optional)

For environment-specific deployments:

1. Go to **Settings** → **Environments**
2. Create environments: `dev`, `staging`, `prod`
3. Add environment-specific secrets if needed
4. Configure protection rules (e.g., require approval for `prod`)

## Usage

### Automatic Deployment

Push to `main` branch to trigger automatic deployment:

```bash
git push origin main
```

### Manual Deployment

1. Go to **Actions** tab in GitHub
2. Select **CD - Deploy to Databricks** workflow
3. Click **Run workflow**
4. Choose environment (dev/staging/prod)
5. Click **Run workflow**

### Monitoring

- View workflow runs in the **Actions** tab
- Check deployment summary in the workflow run page
- Monitor Databricks job execution in the Databricks UI

## Customization

### Modifying CI Checks

Edit `.github/workflows/ci.yml` to:
- Add/remove linting rules
- Configure test coverage requirements
- Add additional validation steps

### Modifying Deployment

Edit `.github/workflows/cd.yml` to:
- Change deployment paths
- Add pre/post-deployment steps
- Configure different environments
- Add notification steps (Slack, email, etc.)

## Troubleshooting

### Deployment Fails

1. Check Databricks token is valid and has correct permissions
2. Verify workspace paths exist
3. Check job name matches in Databricks

### Wheel Upload Fails

1. Ensure DBFS path `/FileStore/wheels/` exists
2. Check token has DBFS write permissions

### Job Trigger Fails

1. Verify job name `arxiv-agent-register-deploy-pipeline` exists in Databricks
2. Check token has job run permissions
3. Install `jq` if running locally: `sudo apt-get install jq`
