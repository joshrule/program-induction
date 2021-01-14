use std::time::Instant;

pub struct Control {
    pub steps: usize,
    pub runtime: usize,
    pub burn: usize,
    pub thin: usize,
    pub restart: usize,
    pub print: usize,
    pub start: Instant,
    pub done_steps: usize,
}

impl Control {
    pub fn new(
        steps: usize,
        runtime: usize,
        burn: usize,
        thin: usize,
        restart: usize,
        print: usize,
    ) -> Self {
        Control {
            steps,
            runtime,
            burn,
            thin,
            restart,
            print,
            done_steps: 0,
            start: Instant::now(),
        }
    }
    pub fn start(&mut self) {
        self.start = Instant::now();
        self.done_steps = 0;
    }
    pub fn running(&mut self) -> bool {
        self.done_steps += 1;
        !((self.steps > 0 && self.done_steps >= self.steps)
            || (self.runtime > 0 && self.start.elapsed().as_millis() as usize >= self.runtime))
    }
}
