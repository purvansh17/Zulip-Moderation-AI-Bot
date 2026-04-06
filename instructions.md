# Install Terraform
brew install terraform   # macOS; or download from terraform.io

# Install Ansible + required collections
pip install ansible
ansible-galaxy collection install community.general ansible.posix
# Add to PATH if required

# Install OpenStack CLI (for creating object store containers)
pip install python-openstackclient

# Confirm your Chameleon SSH key exists
ls ~/.ssh/id_rsa_chameleon   # if named differently, update var.key in terraform

OpenStack credentials (~/.config/openstack/clouds.yaml):
Download from the Chameleon dashboard → Identity → Application Credentials → Download clouds.yaml and place it at ~/.config/openstack/clouds.yaml.

Assume CHI@TACC bare metal instance 

Phase 1 — Terraform (provision VM)
Run from infra/terraform/ on your local machine:


cd infra/terraform
terraform init

# Plan first to review what will be created
terraform plan \
  -var="suffix=proj09" \
  -var="reservation_id=xxxxxxxx-60d5-xxxx-ba6a-bb733aec98e7"

# Apply (will prompt for confirmation)
terraform apply \
  -var="suffix=proj09" \
  -var="reservation_id=xxxxxxxx-60d5-xxxx-ba6a-bb733aec98e7"

# Save the floating IP — you'll need it for every Ansible run
export ANSIBLE_HOST=$(terraform output -raw floating_ip_out)
echo $ANSIBLE_HOST   # e.g. 129.114.x.x


Phase 2 — Secrets (one-time setup)
Run from infra/ansible/ on your local machine:


cd infra/ansible

# Copy the template
cp group_vars/all/vault.yml.example group_vars/all/vault.yml

# Edit vault.yml — fill in ALL values
nano group_vars/all/vault.yml
Key values to fill in:

vault_zulip_secret_key → generate with: python3 -c "import secrets; print(secrets.token_hex(32))"
vault_chameleon_ec2_access / vault_chameleon_ec2_secret → from Chameleon dashboard: Identity → EC2 Credentials → Create EC2 Credential

# Encrypt vault.yml (you'll set a vault password — remember it!)
ansible-vault encrypt group_vars/all/vault.yml

# Store vault password so you don't have to type it each time
echo "your_vault_password" > ~/.vault_pass
chmod 600 ~/.vault_pass

# Disables firewall, installs Docker, creates local storage directory.


ansible-playbook -i inventory.yml pre_k8s.yml \
  --vault-password-file ~/.vault_pass
Verify:


ssh -i ~/.ssh/id_rsa_chameleon cc@$ANSIBLE_HOST \
  "docker --version && docker ps"

Installs k3s, configures NVIDIA for k3s containerd, deploys nginx-ingress.


ansible-playbook -i inventory.yml install_k3s.yml \
  --vault-password-file ~/.vault_pass
Verify:


ssh -i ~/.ssh/id_rsa_chameleon cc@$ANSIBLE_HOST \
  "kubectl get nodes -o wide && nvidia-smi"


Creates Swift containers (locally), copies manifests, creates K8s secrets, deploys all services.


ansible-playbook -i inventory.yml post_k8s.yml \
  --vault-password-file ~/.vault_pass
At the end it prints your access URLs:


Zulip:  http://zulip.<IP>.nip.io
MLflow: http://mlflow.<IP>.nip.io

