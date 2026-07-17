#pragma once

#include <QImage>
#include <QLabel>

class QResizeEvent;

// Displays the latest processed video frame, scaled to fit while preserving
// aspect ratio. Meant to sit as the large, central widget of the window.
class VideoView : public QLabel
{
    Q_OBJECT

public:
    explicit VideoView(QWidget *parent = nullptr);

public slots:
    void setFrame(const QImage &image);

protected:
    void resizeEvent(QResizeEvent *event) override;

private:
    void updatePixmap();

    QImage m_lastFrame;
};
