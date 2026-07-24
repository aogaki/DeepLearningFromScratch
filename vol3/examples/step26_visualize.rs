use std::fs::File;
use std::io::Write;
use std::process::Command;
use vol3::utils::get_dot_graph;
use vol3::variable::Variable;

fn main() {
    // Construct a sample computational graph
    let x0 = Variable::from(1.0);
    x0.set_name("x0");

    let x1 = Variable::from(1.0);
    x1.set_name("x1");

    let y = &x0 + &x1;
    y.set_name("y");

    let z = y.square();
    z.set_name("z");

    let txt = get_dot_graph(&z, true);

    println!("Graph generated!");

    // Write dot file
    let output_dir = "output";
    std::fs::create_dir_all(output_dir).unwrap();
    let dot_path = format!("{}/sample.dot", output_dir);
    let png_path = format!("{}/sample.png", output_dir);

    if let Ok(mut file) = File::create(&dot_path)
        && file.write_all(txt.as_bytes()).is_ok()
    {
        println!("Saved DOT to {}", dot_path);

        // Execute dot command
        match Command::new("dot")
            .args(["-Tpng", &dot_path, "-o", &png_path])
            .status()
        {
            Ok(status) if status.success() => {
                println!("Successfully generated PNG: {}", png_path);
            }
            Ok(status) => {
                eprintln!("dot command failed with exit status: {}", status);
            }
            Err(e) => {
                eprintln!("Failed to run dot command: {}", e);
            }
        }
    }
}
