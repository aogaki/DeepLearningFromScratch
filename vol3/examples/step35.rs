use std::process::Command;
use vol3::utils::get_dot_graph;
use vol3::variable::Variable;

fn main() {
    let x = Variable::from(1.0);
    x.set_name("x");

    let mut y = x.tanh();
    y.set_name("y");

    let iters = 6; // 6階微分まで
    for i in 0..iters {
        x.cleargrad();
        y.backward(false, true); // create_graph=true
        y = x.grad_var().unwrap();
        y.set_name(&format!("gx{}", i + 1));
    }

    let dot_content = get_dot_graph(&y, false);
    let output_dir = "output";
    std::fs::create_dir_all(output_dir).unwrap();
    let dot_path = "output/step35.dot";
    let png_path = "output/step35.png";
    std::fs::write(dot_path, dot_content).expect("Unable to write dot file");

    let output = Command::new("dot")
        .args(["-Tpng", dot_path, "-o", png_path])
        .output();

    match output {
        Ok(out) if out.status.success() => println!("Generated {}", png_path),
        _ => println!("dot command failed, is graphviz installed?"),
    }
}
