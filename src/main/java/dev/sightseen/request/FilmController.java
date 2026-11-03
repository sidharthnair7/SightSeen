package dev.sightseen.request;

import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

/**
 * The landlord's side, reached only through the link the student sent. It shows which place the video is for and
 * takes the video. Nothing about the student or the 3D room comes back.
 */
@RestController
@RequestMapping("/api/film")
class FilmController {

    record FilmView(String label, boolean videoReceived) {
        static FilmView of(RoomRequest r) {
            return new FilmView(r.getLabel(), r.hasVideo());
        }
    }

    private final RequestService service;

    FilmController(RequestService service) {
        this.service = service;
    }

    @GetMapping("/{token}")
    FilmView get(@PathVariable String token) {
        return FilmView.of(service.byLandlordToken(token));
    }

    @PostMapping(path = "/{token}/video", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    FilmView upload(@PathVariable String token, @RequestParam("video") MultipartFile video,
                    @RequestParam(name = "live", defaultValue = "false") boolean live) {
        return FilmView.of(service.receiveVideo(service.byLandlordToken(token), video, VideoSource.LANDLORD_LINK, live));
    }
}
