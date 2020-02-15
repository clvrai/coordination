
----

## Our method and baselines

![Baseline illustrations](./img/baselines.png "")

For a cooperative task that requires N agents or end-effectors to work together, we propose a modular framework with skill behavior diversification (<b>Modular-SBD</b>), which first individually trains each agent's primitive skills with diverse behaviors conditioned on a <i>behavior embedding</i> z. Then, a meta policy takes as input the full observation and selects both a primitive skill and a behavior embedding for each agent.
<br/>
To compare the performance of our method with various single- and multi-agent RL methods, we designed 5 baselines illustrated in the figure above. The <b>RL</b> and <b>MARL</b> baselines are vanilla RL frameworks that are widely used in single- and multi-agent RL literature. The <b>Modular</b> baseline is a hierarchical framework that composes of a meta policy and N sets of primitive skills and the meta policy selects a primitive skill to execute for each agent but not the behavior of the skill. In addition, we also consider the <b>RL-SBD</b> and <b>MARL-SBD</b> baselines. These baselines are the RL and MARL baselines augmented by a meta policy that outputs skill behavior embeddings served as an additional input to low-level policies.

----

## Videos

<span class="env-name"><b>Jaco Pick-Push-Place</b></span>
- Two Jaco arms start with a block on the left and a container on the right. To complete the task, the arms need to pick up the block, push the container to the center, and place the block inside the container.
<div class="w3-row-padding">
	<div class="w3-col s4 w3-center">
		<video height="auto" width="100%" controls>
		  <source src="./video/ppp_single.mp4" type="video/mp4">
		</video>
		<div class="method-name">RL</div>
	</div>
	<div class="w3-col s4 w3-center">
		<video height="auto" width="100%" controls>
		  <source src="./video/ppp_nodiayn.mp4" type="video/mp4">
		</video>
		<div class="method-name">Modular</div>
	</div>
	<div class="w3-col s4 w3-center">
		<video height="auto" width="100%" controls>
		  <source src="./video/ppp_ours.mp4" type="video/mp4">
		</video>
		<div class="method-name">Modular with SBD (Ours)</div>
	</div>
</div>
<span class="env-name"><b>Jaco Pick-Move-Place</b></span>
- Two Jaco arms need to pick up a long bar together, move the bar towards a target location while maintaining its rotation, and place it on the table.
<div class="w3-row-padding">
	<div class="w3-col s4 w3-center">
		<video height="auto" width="100%" controls>
		  <source src="./video/pmp_single.mp4" type="video/mp4">
		</video>
		<div class="method-name">RL</div>
	</div>
	<div class="w3-col s4 w3-center">
		<video height="auto" width="100%" controls>
		  <source src="./video/pmp_nodiayn.mp4" type="video/mp4">
		</video>
		<div class="method-name">Modular</div>
	</div>
	<div class="w3-col s4 w3-center">
		<video height="auto" width="100%" controls>
		  <source src="./video/pmp_ours.mp4" type="video/mp4">
		</video>
		<div class="method-name">Modular with SBD (Ours)</div>
	</div>
</div>
<span class="env-name"><b>Two Ant Push</b></span>
- Two ants need to push a large object toward a green target place, collaborating with each other to keep the angle of the object as stable as possible.
<div class="w3-row-padding">
	<div class="w3-col s4 w3-center">
		<video height="auto" width="100%" controls>
		  <source src="./video/ant_push_single.mp4" type="video/mp4">
		</video>
		<div class="method-name">RL</div>
	</div>
	<div class="w3-col s4 w3-center">
		<video height="auto" width="100%" controls>
		  <source src="./video/ant_push_nodiayn.mp4" type="video/mp4">
		</video>
		<div class="method-name">Modular</div>
	</div>
	<div class="w3-col s4 w3-center">
		<video height="auto" width="100%" controls>
		  <source src="./video/ant_push_ours.mp4" type="video/mp4">
		</video>
		<div class="method-name">Modular with SBD (Ours)</div>
	</div>
</div>

----

## Quantitative results

![Learning curves](./img/training.png "")

![Success rates](./img/table.png "Table")

----

## Citation
```
@inproceedings{lee2020learning,
  title={Learning to Coordinate Manipulation Skills via Skill Behavior Diversification},
  author={Youngwoon Lee and Jingyun Yang and Joseph J. Lim},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=ryxB2lBtvH}
}
```
