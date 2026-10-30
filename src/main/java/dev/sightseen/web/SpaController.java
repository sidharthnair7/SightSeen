package dev.sightseen.web;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

/** React draws these pages; the server only hands over index.html (from frontend/dist). */
@Controller
class SpaController {

    @GetMapping({"/film/{token}", "/room/{id}"})
    String page() {
        return "forward:/index.html";
    }
}
