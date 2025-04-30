from workers.actor import Actor
from workers.critic import Critic
from workers.ref import Ref
from ppo import PPO


actor = Actor()
critic = Critic()
ref = Ref()
ppo = PPO(actor, critic, ref)

# Main training loop
def main():
    for epoch in range(NUM_EPOCHS):
        # Step 1: Generate rollouts using the actor
        rollouts = actor.generate_rollouts()

        # Step 2: Compute rewards using the critic
        rewards = critic.compute_rewards(rollouts)

        # Step 3: Compute KL divergence with the reference model
        kl_divergence = ref.compute_kl(rollouts)

        # Step 4: Perform PPO optimization
        ppo.optimize(rollouts, rewards, kl_divergence)

        # Step 5: Log training progress
        log_metrics(epoch, rewards, kl_divergence)

if __name__ == "__main__":
    main()